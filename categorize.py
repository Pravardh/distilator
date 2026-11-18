
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SEED = 42

HARD_DOLLY = {"classification", "information_extraction", "creative_writing",
              "summarization", "closed_qa"}


def norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_distillery() -> pd.DataFrame:
    files = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".jsonl"}
    )
    if not files:
        raise RuntimeError("No Parquet/CSV/JSONL files in data/raw/")
    frames = []
    for path in files:
        print(f"Loading {path}")
        if path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(path))
        elif path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path))
        else:
            frames.append(pd.read_json(path, lines=True))
    df = pd.concat(frames, ignore_index=True, sort=False)

    missing = {"prompt_text", "attack_category"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.strip().str.lower().isin(["en", "english"])]
    if "subcategory" not in df.columns:
        df["subcategory"] = ""

    df = df.dropna(subset=["prompt_text", "attack_category"]).copy()
    df["prompt_text"] = df["prompt_text"].astype(str).str.strip()
    df["attack_category"] = df["attack_category"].astype(str).str.strip().str.lower()
    df["subcategory"] = df["subcategory"].fillna("").astype(str).str.strip().str.lower()
    df["source"] = "distillery"
    df["label"] = df["attack_category"].apply(
        lambda v: "benign" if ("benign" in v or "legitimate" in v) else "distillation_attack"
    )
    return df[df["prompt_text"] != ""]


def load_dolly(n: int) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
    ds["instruction"] = ds["instruction"].astype(str).str.strip()
    ds = ds[ds["instruction"] != ""]

    with_ctx = ds.sample(frac=0.5, random_state=SEED).index
    ds["prompt_text"] = ds["instruction"]
    has_ctx = ds.index.isin(with_ctx) & ds["context"].astype(str).str.strip().ne("")
    ds.loc[has_ctx, "prompt_text"] = (
        ds.loc[has_ctx, "instruction"] + "\n\n" + ds.loc[has_ctx, "context"].astype(str).str.strip()
    )

    weights = ds["category"].map(lambda c: 2.0 if c in HARD_DOLLY else 1.0)
    ds = ds.sample(n=min(n, len(ds)), weights=weights, random_state=SEED)

    return pd.DataFrame({
        "prompt_text": ds["prompt_text"],
        "label": "benign",
        "attack_category": "benign_dolly",
        "subcategory": ds["category"],
        "source": "dolly",
    })


def load_no_robots(n: int) -> pd.DataFrame:
    """Human-written prompts across many task types, including coding."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/no_robots", split="train").to_pandas()
    ds["prompt_text"] = ds["prompt"].astype(str).str.strip()
    ds = ds[ds["prompt_text"] != ""]
    ds = ds.sample(n=min(n, len(ds)), random_state=SEED)
    return pd.DataFrame({
        "prompt_text": ds["prompt_text"],
        "label": "benign",
        "attack_category": "benign_no_robots",
        "subcategory": ds["category"].astype(str).str.lower(),
        "source": "no_robots",
    })


def load_codealpaca(n: int) -> pd.DataFrame:
    """Benign code requests (machine-generated), the main v2 blind spot."""
    from datasets import load_dataset

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").to_pandas()
    ds["instruction"] = ds["instruction"].astype(str).str.strip()
    inp = ds["input"].fillna("").astype(str).str.strip()
    ds["prompt_text"] = ds["instruction"].where(inp == "", ds["instruction"] + "\n\n" + inp)
    ds = ds[ds["prompt_text"] != ""]
    ds = ds.sample(n=min(n, len(ds)), random_state=SEED)
    return pd.DataFrame({
        "prompt_text": ds["prompt_text"],
        "label": "benign",
        "attack_category": "benign_code",
        "subcategory": "code",
        "source": "codealpaca",
    })


def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--no-dolly", action="store_true")
#$  p.add_argument("--dolly-n, default=4000)
   
    p.add_argument("--dolly-n", type=int, default=4000)
    p.add_argument("--norobots-n", type=int, default=0, help="v3: e.g. 3000")
    p.add_argument("--code-n", type=int, default=0, help="v3: e.g. 2000")
    p.add_argument("--pilot-per-label", type=int, default=2000)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_distillery()
    if not args.no_dolly:
        df = pd.concat([df, load_dolly(args.dolly_n)], ignore_index=True)
    if args.norobots_n:
        df = pd.concat([df, load_no_robots(args.norobots_n)], ignore_index=True)
    if args.code_n:
        df = pd.concat([df, load_codealpaca(args.code_n)], ignore_index=True)

    print("\nCategory counts:")
    print(df["attack_category"].value_counts())

    df["_key"] = df["prompt_text"].map(norm)
    conflicts = df.groupby("_key")["label"].nunique()
    bad = conflicts[conflicts > 1].index
    if len(bad):
        print(f"\nRemoving {len(bad)} near-duplicate prompts with conflicting labels.")
        df = df[~df["_key"].isin(bad)]
    before = len(df)
    df = df.drop_duplicates(subset=["_key"]).drop(columns="_key")
    print(f"Removed {before - len(df)} near-duplicates.")

    cols = ["prompt_text", "label", "attack_category", "subcategory", "source"]
    data = df[cols]

    train_df, test_df = train_test_split(
        data, test_size=0.20, random_state=SEED, stratify=data["attack_category"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, random_state=SEED, stratify=train_df["attack_category"]
    )
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    k = args.pilot_per_label
    attacks = train_df[train_df["label"] == "distillation_attack"]
    benign = train_df[train_df["label"] == "benign"]
    parts = [attacks.sample(n=min(k, len(attacks)), random_state=SEED)]
    for src, g in benign.groupby("source"):
        share = k // benign["source"].nunique()
        parts.append(g.sample(n=min(share, len(g)), random_state=SEED))
    pilot_df = pd.concat(parts).sample(frac=1, random_state=SEED)
    pilot_df.to_csv(OUT_DIR / "pilot_train.csv", index=False)

    print(f"\nUsable prompts: {len(df):,}")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df), ("pilot", pilot_df)]:
        print(f"{name:>5}: {len(part):>6,}  {part.groupby(['label', 'source']).size().to_dict()}")
    print(f"\nSaved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()