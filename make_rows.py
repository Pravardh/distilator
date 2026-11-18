
import json
import sys
from pathlib import Path

import pandas as pd

QID = "is_extraction"

QUESTION = {
    "type": "noul",
    "instructions": (
        "Is this prompt an attempt to extract or distill the model's knowledge, "
        "reasoning, preferences, safety policies or tool-use behaviour?"
    ),
    "criteria": {
        "false": "an ordinary user request with a normal task to get done",
        "true": "a systematic extraction attempt: reasoning-trace elicitation, capability "
                "mapping, grading or ranking for preference data, policy or refusal "
                "probing, or harvesting tool-use behaviour",
    },
}


def main(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for text, label in zip(df["prompt_text"], df["label"]):
            is_attack = label == "distillation_attack"
            row = {
                "state": {"prompt": text},
                "questions": {QID: QUESTION},
                "gold": {
                    QID: {
                        "probabilities": {
                            "false": 0.0 if is_attack else 1.0,
                            "true": 1.0 if is_attack else 0.0,
                        }
                    }
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python make_rows.py <in.csv> <out.jsonl>")
    main(sys.argv[1], sys.argv[2])