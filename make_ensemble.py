
import re
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

N_PROBES = 33


def probe_scores(tag):
    path = Path(f"logs/probes/probes_{tag}.log")
    if not path.exists():
        sys.exit(f"missing {path}")
    s = re.findall(r"Laya\s+P\(attack\) = ([\d.]+)", path.read_text())
    if len(s) < N_PROBES:
        sys.exit(f"{path}: expected {N_PROBES} scores, found {len(s)}")
    return [float(x) for x in s[:N_PROBES]]


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: python make_ensemble.py OUT_TAG TAG1 TAG2 [TAG3 ...]")
    out, tags = sys.argv[1], sys.argv[2:]

    # Probes: mean score per line.
    per_model = [probe_scores(t) for t in tags]
    avg = [sum(v) / len(v) for v in zip(*per_model)]
    lines = [f"Ensemble of {', '.join(tags)} (mean Laya score)\n\n"]
    for i, (s, parts) in enumerate(zip(avg, zip(*per_model)), 1):
        detail = "  ".join(f"{t}={p:.3f}" for t, p in zip(tags, parts))
        lines.append(f"probe line {i}   [{detail}]\n  Laya     P(attack) = {s:.3f}\n\n")
    Path(f"logs/probes/probes_{out}.log").write_text("".join(lines))
    print(f"wrote probes_{out}.log ({len(avg)} probes)")

    # Real-traffic sets: mean score per prompt, matched on prompt text.
    for src in ("wildchat", "oasst", "hh"):
        files = [Path(f"results/{src}_scores_{t}.csv") for t in tags]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            print(f"skip {src}: missing {', '.join(missing)}")
            continue
        dfs = []
        for t, f in zip(tags, files):
            d = pd.read_csv(f)[["category", "prompt_text", "laya_score"]]
            d = d.drop_duplicates("prompt_text").rename(columns={"laya_score": f"s_{t}"})
            dfs.append(d)
        merged = reduce(
            lambda a, b: a.merge(b.drop(columns="category"), on="prompt_text", how="inner"), dfs)
        sizes = [len(d) for d in dfs]
        if len(merged) != min(sizes):
            print(f"WARNING {src}: inputs had {sizes} prompts but only {len(merged)} matched; "
                  f"the runs may have scored different samples")
        merged["laya_score"] = merged[[f"s_{t}" for t in tags]].mean(axis=1)
        merged.to_csv(f"results/{src}_scores_{out}.csv", index=False)
        print(f"wrote results/{src}_scores_{out}.csv ({len(merged):,} prompts; "
              f"FPR @0.5 {(merged['laya_score'] >= 0.5).mean():.2%})")


if __name__ == "__main__":
    main()
