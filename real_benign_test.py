
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import laya
from make_rows import QID, QUESTION

QUESTIONS = {QID: QUESTION}
THRESHOLDS = [0.5, 0.7, 0.9]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:6.2%}  [{lo:6.2%} - {hi:6.2%}]  ({k}/{n})"


def load_oasst(n, seed):
    ds = load_dataset("OpenAssistant/oasst1", split="train").to_pandas()
    ds = ds[(ds["role"] == "prompter") & (ds["parent_id"].isna()) & (ds["lang"] == "en")]
    ds = ds.rename(columns={"text": "prompt_text"})
    ds["prompt_text"] = ds["prompt_text"].astype(str).str.strip()
    ds = ds[ds["prompt_text"] != ""].drop_duplicates("prompt_text")
    ds["category"] = "oasst"
    return ds.sample(n=min(n, len(ds)), random_state=seed).reset_index(drop=True)


def load_hh(n, seed):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train").to_pandas()

    def first_human(text):
        text = str(text)
        start = text.find("Human:")
        if start == -1:
            return ""
        rest = text[start + len("Human:"):]
        end = rest.find("\n\nAssistant:")
        return (rest if end == -1 else rest[:end]).strip()

    ds["prompt_text"] = ds["chosen"].map(first_human)
    ds = ds[ds["prompt_text"] != ""].drop_duplicates("prompt_text")
    ds["category"] = "hh_helpful"
    return ds[["prompt_text", "category"]].sample(n=min(n, len(ds)), random_state=seed).reset_index(drop=True)


def load_wildchat(n, seed, pool=40_000):
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rows = []
    for ex in ds:
        if ex.get("language") != "English" or ex.get("toxic"):
            continue
        conv = ex.get("conversation") or []
        first = next((t.get("content", "") for t in conv if t.get("role") == "user"), "")
        first = str(first).strip()
        if first:
            rows.append(first)
        if len(rows) >= pool:
            break
    df = pd.DataFrame({"prompt_text": rows}).drop_duplicates("prompt_text")
    df["category"] = "wildchat"
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def load_dolly(n, with_context, seed):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
    if with_context:
        ds["prompt_text"] = ds.apply(
            lambda r: f"{r['instruction']}\n\n{r['context']}".strip() if r["context"] else r["instruction"],
            axis=1,
        )
    else:
        ds["prompt_text"] = ds["instruction"]
    ds["prompt_text"] = ds["prompt_text"].astype(str).str.strip()
    ds = ds[ds["prompt_text"] != ""].drop_duplicates("prompt_text")
    return ds.sample(n=min(n, len(ds)), random_state=seed).reset_index(drop=True)


def laya_scores(agent, texts, batch_size=32):
    states = [{"prompt": t} for t in texts]
    res = agent.predict_batch(states, QUESTIONS, batch_size=batch_size, sort_by_length=True)
    return np.array([r["answers"][QID]["noul"] for r in res], dtype=float)


def tfidf_scores(train_csv, texts):
    tr = pd.read_csv(train_csv)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(tr["prompt_text"]), (tr["label"] == "distillation_attack").astype(int))
    return clf.predict_proba(vec.transform(texts))[:, 1]


def report(name, scores, df):
    print(f"\n=== {name} (all benign: every flag is a false positive) ===")
    n = len(scores)
    for th in THRESHOLDS:
        print(f"  FPR @ {th}: {fmt_rate(int((scores >= th).sum()), n)}")
    print("  score distribution: "
          + "  ".join(f"<{q}: {(scores < q).mean():.1%}" for q in [0.1, 0.5, 0.9]))
    print("  FPR @ 0.5 by category:")
    for cat, g in df.assign(s=scores).groupby("category"):
        k = int((g["s"] >= 0.5).sum())
        print(f"    {cat:<24} {fmt_rate(k, len(g))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", default="oasst", choices=["oasst", "dolly", "hh", "wildchat"],
                   help="use oasst for any model trained with Dolly in its benign class")
    p.add_argument("--tag", default="", help="suffix for output files, e.g. v1 or v2")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--with-context", action="store_true",
                   help="append Dolly's reference passage, like a real user pasting text")
    p.add_argument("--tfidf-train", default="data/processed/pilot_train.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.source == "oasst":
        df = load_oasst(args.n, args.seed)
    elif args.source == "hh":
        df = load_hh(args.n, args.seed)
    elif args.source == "wildchat":
        df = load_wildchat(args.n, args.seed, pool=max(args.n * 3, 10_000))
    else:
        df = load_dolly(args.n, args.with_context, args.seed)
    texts = df["prompt_text"].tolist()
    print(f"Loaded {len(df):,} {args.source} prompts")

    agent = laya.load(args.model)
    df["laya_score"] = laya_scores(agent, texts)
    del agent
    torch.cuda.empty_cache()
    report("laya_finetuned", df["laya_score"].to_numpy(), df)

    df["tfidf_score"] = tfidf_scores(args.tfidf_train, texts)
    report("tfidf_logreg", df["tfidf_score"].to_numpy(), df)

    out = Path("results")
    out.mkdir(exist_ok=True)
    suffix = ("_ctx" if args.with_context and args.source == "dolly" else "") + (
        f"_{args.tag}" if args.tag else "")
    base = f"{args.source}"
    df[["category", "prompt_text", "laya_score", "tfidf_score"]].to_csv(
        out / f"{base}_scores{suffix}.csv", index=False)
    flagged = df[df["laya_score"] >= 0.5].sort_values("laya_score", ascending=False)
    flagged[["laya_score", "tfidf_score", "category", "prompt_text"]].to_csv(
        out / f"{base}_flagged{suffix}.csv", index=False)

    print(f"\nTop 15 flagged by Laya (of {len(flagged)}):")
    for _, r in flagged.head(15).iterrows():
        print(f"  {r['laya_score']:.3f} | {r['category']:<22} | {r['prompt_text'][:140]}")
    print(f"\nSaved results/{base}_scores{suffix}.csv and results/{base}_flagged{suffix}.csv")


if __name__ == "__main__":
    main()