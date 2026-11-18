
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import laya
from make_rows import QID, QUESTION  # identical question wording to training

QUESTIONS = {QID: QUESTION}
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
PREVALENCES = [0.01, 0.001]  # realistic share of extraction traffic in production


def laya_scores(agent, texts, batch_size=32):
    states = [{"prompt": t} for t in texts]
    results = agent.predict_batch(states, QUESTIONS, batch_size=batch_size, sort_by_length=True)
    return np.array([r["answers"][QID]["noul"] for r in results], dtype=float)


def laya_latency(agent, texts, n=100):
    sample = texts[:n]
    agent.predict({"prompt": sample[0]}, QUESTIONS)  # warm-up
    times = []
    for t in sample:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        agent.predict({"prompt": t}, QUESTIONS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"p50_ms": statistics.median(times), "p95_ms": times[int(0.95 * (len(times) - 1))]}


def tfidf_scores(train_df, texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train_df["prompt_text"])
    y = (train_df["label"] == "distillation_attack").astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    start = time.perf_counter()
    scores = clf.predict_proba(vec.transform(texts))[:, 1]
    per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
    return scores, {"p50_ms": per_item_ms, "p95_ms": None}


def precision_at_prevalence(tpr, fpr, prevalence):
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else float("nan")


def evaluate(name, y, scores, df):
    out = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "thresholds": {},
    }
    for th in THRESHOLDS:
        pred = (scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        out["thresholds"][th] = {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "fpr": float(fpr),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "precision_at_prevalence": {
                str(p): precision_at_prevalence(tpr, fpr, p) for p in PREVALENCES
            },
        }

    # Per-category: recall for attack categories, false-positive rate for benign.
    pred = (scores >= 0.5).astype(int)
    per_cat = {}
    for cat, idx in df.groupby("attack_category").groups.items():
        idx = df.index.get_indexer(idx)
        if "benign" in cat or "legitimate" in cat:
            per_cat[cat] = {"n": len(idx), "false_positive_rate": float(pred[idx].mean())}
        else:
            per_cat[cat] = {"n": len(idx), "recall": float(pred[idx].mean())}
    out["per_category@0.5"] = per_cat
    return out


def print_report(name, r, latency):
    print(f"\n=== {name} ===")
    print(f"ROC-AUC {r['roc_auc']:.4f}   PR-AUC {r['pr_auc']:.4f}   "
          f"latency p50 {latency['p50_ms']:.2f} ms"
          + (f"  p95 {latency['p95_ms']:.2f} ms" if latency.get("p95_ms") else ""))
    print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'f1':>7} {'fpr':>7} {'prec@1%':>9} {'prec@0.1%':>10}")
    for th, m in r["thresholds"].items():
        pp = m["precision_at_prevalence"]
        print(f"{th:>5} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['fpr']:>7.4f} {pp['0.01']:>9.3f} {pp['0.001']:>10.3f}")
    print("per category @0.5:")
    for cat, m in sorted(r["per_category@0.5"].items()):
        stat = (f"FPR {m['false_positive_rate']:.3f}" if "false_positive_rate" in m
                else f"recall {m['recall']:.3f}")
        print(f"  {cat:<26} n={m['n']:<5} {stat}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="fine-tuned model dir")
    p.add_argument("--base", default="./laya_base", help="base checkpoint dir")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--tfidf-train", default="data/processed/pilot_train.csv",
                   help="defaults to the pilot set so the comparison uses the same data")
    p.add_argument("--latency-n", type=int, default=100)
    p.add_argument("--tag", default="", help="suffix for output files, e.g. v1 or v2")
    args = p.parse_args()

    df = pd.read_csv(f"data/processed/{args.split}.csv").reset_index(drop=True)
    texts = df["prompt_text"].tolist()
    y = (df["label"] == "distillation_attack").astype(int).to_numpy()
    print(f"{args.split}: {len(df):,} prompts, {y.sum():,} attacks, {len(y) - y.sum():,} benign")

    results, preds = {}, df[["prompt_text", "label", "attack_category", "subcategory"]].copy()

    for name, path in [("laya_base_zero_shot", args.base), ("laya_finetuned", args.model)]:
        print(f"\nScoring with {name} ({path}) ...")
        agent = laya.load(path)
        scores = laya_scores(agent, texts)
        lat = laya_latency(agent, texts, args.latency_n)
        results[name] = {**evaluate(name, y, scores, df), "latency": lat}
        preds[f"{name}_score"] = scores
        print_report(name, results[name], lat)
        del agent
        torch.cuda.empty_cache()

    print("\nTraining TF-IDF + LogReg baseline ...")
    scores, lat = tfidf_scores(pd.read_csv(args.tfidf_train), texts)
    results["tfidf_logreg"] = {**evaluate("tfidf_logreg", y, scores, df), "latency": lat}
    preds["tfidf_logreg_score"] = scores
    print_report("tfidf_logreg", results["tfidf_logreg"], lat)

    out = Path("results")
    out.mkdir(exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    (out / f"{args.split}_results{sfx}.json").write_text(json.dumps(results, indent=2))
    preds.to_csv(out / f"{args.split}_predictions{sfx}.csv", index=False)
    print(f"\nSaved results/{args.split}_results{sfx}.json and results/{args.split}_predictions{sfx}.csv")


if __name__ == "__main__":
    main()
