
import argparse
import json
from pathlib import Path

import pandas as pd

SEED = 42
COLS = ["prompt_text", "label", "attack_category", "subcategory", "source"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aug", nargs="+", default=["data/augmented/aug.jsonl"],
                   help="one or more rewrite files, e.g. aug.jsonl aug_content.jsonl")
    p.add_argument("--base", default="data/processed/pilot_train.csv")
    p.add_argument("--total", type=int, default=3000)
    p.add_argument("--aug-share", type=float, default=0.5, help="share of rewrites per label")
    p.add_argument("--out", default="data/processed_v4/pilot_train.csv")
    args = p.parse_args()

    rows = [json.loads(l) for path in args.aug for l in open(path, encoding="utf-8") if l.strip()]
    aug = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    print(f"Rewrites: {len(rows):,} total, {len(aug):,} accepted")
    print(pd.Series([r["status"] for r in rows]).value_counts().to_string(), "\n")

    aug = pd.DataFrame({
        "prompt_text": aug["prompt_text"],
        "label": aug["label"],
        "attack_category": aug["category"],
        "subcategory": aug["subcategory"].fillna(""),
        "source": "aug_" + aug["style"],
    }).drop_duplicates("prompt_text")

    base = pd.read_csv(args.base)
    for c in COLS:
        if c not in base.columns:
            base[c] = ""
    base = base[COLS]

    per_label = args.total // 2
    parts = []
    for label in ["distillation_attack", "benign"]:
        a = aug[aug["label"] == label]
        n_aug = min(int(per_label * args.aug_share), len(a))
        b = base[base["label"] == label]
        n_base = min(per_label - n_aug, len(b))
        parts += [a.sample(n=n_aug, random_state=SEED), b.sample(n=n_base, random_state=SEED)]
        print(f"{label:<20} original {n_base:>5}   rewrites {n_aug:>5}")

    out_df = pd.concat(parts).sample(frac=1, random_state=SEED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df):,} rows to {args.out}")
    print(out_df.groupby(["label", "source"]).size().to_string())


if __name__ == "__main__":
    main()
