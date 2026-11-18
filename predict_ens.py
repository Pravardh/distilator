
import argparse
import sys
import time
import warnings

import laya
from make_rows import QID, QUESTION

QUESTIONS = {QID: QUESTION}
DEFAULT_MODELS = [
    "models/distilguard-v4",    # seed 42
    "models/distilguard-v4s2",  # seed 63
    "models/distilguard-v4s3",  # seed 7
]
DEFAULT_THR = 0.485  


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="prompt to score; omit for interactive / stdin")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--thr", type=float, default=DEFAULT_THR)
    p.add_argument("--device", default=None, help="cpu or cuda")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="laya")
    agents = []
    for m in args.models:
        print(f"Loading {m} ...", file=sys.stderr)
        agents.append(laya.load(m, device=args.device) if args.device else laya.load(m))
    names = [m.rstrip("/").rsplit("-", 1)[-1] for m in args.models]

    def score(text):
        start = time.perf_counter()
        s = [a.predict({"prompt": text}, QUESTIONS)["answers"][QID]["noul"] for a in agents]
        ms = (time.perf_counter() - start) * 1000
        mean = sum(s) / len(s)
        print(text)
        print("  seeds   " + "  ".join(f"{n}={x:.3f}" for n, x in zip(names, s)))
        print(f"  score   {mean:.3f}   ({ms:.0f} ms)")
        print(f"  Verdict: {'SUS' if mean >= args.thr else 'ok'}  "
              f"(mean of {len(s)} seeds >= {args.thr})\n")

    if args.prompt:
        score(args.prompt)
        return
    if sys.stdin.isatty():
        print("Type a prompt and press Enter (Ctrl+D to quit).\n", file=sys.stderr)
    for line in sys.stdin:
        text = line.strip()
        if text:
            score(text)


if __name__ == "__main__":
    main()
