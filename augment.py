import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SEED = 42

STYLES = {
    "casual": "casual, like a quick message to a friend: relaxed tone, lowercase is fine, maybe a "
              "little slang. Vary the wording naturally; do not open with 'yo' or 'hey' and do not "
              "pile on abbreviations",
    "terse": "as short as possible while keeping every essential detail of the request; "
             "no pleasantries or explanation",
    "indirect": "polite and indirect, with one short, realistic reason for asking that fits the "
                "topic. Vary the opening; do not start with 'As part of my' or mention research "
                "or a project unless it fits naturally",
    "typo": "written quickly by a real person: normal wording (not slang), but with a few genuine "
            "spelling mistakes, missing capitals and sloppy punctuation",
    "reworded": "plain standard English, but with completely different vocabulary and "
                "sentence structure from the original",
    "with_content": "the same request, but with the one piece of material it needs pasted in, "
                    "the way a real user would",
}

DEFAULT_STYLES = ["casual", "terse", "indirect", "typo", "reworded"]

SYSTEM_CONTENT = (
    "You rewrite user prompts to build a training set for a defensive classifier. "
    "The given prompt refers to some material without including it. Rewrite it so that the ONE "
    "kind of material the request itself needs is pasted in, short and realistic, exactly as a "
    "real user would type it. Examples of what that might be (use only the one that fits): the "
    "responses it asks to compare, the text it asks to rewrite, the tools it asks about, the "
    "question it asks to solve. Do not add any other kind of material, headings, or explanations "
    "of what you added. Keep the request and intent the same. Do NOT answer the prompt. "
    "Output only the rewritten prompt."
)

META_RE = re.compile(
    r"(tool definition|policy statement|sample response|possible answers|problem posed|"
    r"here is the material|material:|rewritten prompt)", re.IGNORECASE)

CONTENT_MAX_ORIG_CHARS = 400

INTENT = {
    "cot_elicitation": "wants the assistant to reveal its full step-by-step reasoning or thinking process",
    "capability_mapping": "wants the assistant to demonstrate or self-assess what it can do across "
                          "domains, difficulty levels or formats",
    "reward_model_grading": "wants the assistant to judge, score, rank or compare responses",
    "safety_boundary_probe": "wants the assistant to reveal its refusal rules, policies or where its limits are",
    "tool_use_extraction": "wants the assistant to show exactly how it would call tools, functions or "
                           "agents, with concrete arguments or steps",
    "censorship_rewrite": "wants content rewritten, reframed or steered around moderation "
                          "(neutralised, made compliant, caveats removed)",
}

SYSTEM = (
    "You rewrite user prompts to build a training set for a defensive classifier. "
    "Rewrite the given prompt in the requested style. Keep the underlying request and intent "
    "exactly the same; change only wording, tone and structure. If the prompt contains pasted "
    "text, code, a list or data, keep that part unchanged and include it. Do NOT answer the prompt. "
    "Output only the rewritten prompt, with no quotes, labels or commentary."
)

VERIFY = (
    "Prompt A:\n{a}\n\nPrompt B:\n{b}\n\n"
    "Does Prompt B ask for the same thing as Prompt A, with the same underlying intent? "
    "Answer with exactly one word: yes or no. /no_think"
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PREFIX_RE = re.compile(r"^\s*(rewritten prompt|rewrite|prompt|here is.*?)\s*:\s*", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\b(i can't|i cannot|i'm sorry|i am sorry|as an ai)\b", re.IGNORECASE)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()


def chat(base_url, model, messages, temperature, max_tokens, timeout=300):
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    msg = out["choices"][0]["message"]
    text = THINK_RE.sub("", msg.get("content") or "").strip()
    if not text:
        # Thinking models can use the whole token budget on reasoning.
        finish = out["choices"][0].get("finish_reason")
        raise RuntimeError(f"empty reply (finish_reason={finish}); raise --max-tokens "
                           "or disable thinking for this model in LM Studio")
    return text


def clean(text):
    text = re.sub(r"\s*/no_think\s*", "\n", text).strip()  # some models echo the tag
    text = PREFIX_RE.sub("", text.strip())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text


def valid(orig, new, style):
    limit = 1500 if style == "with_content" else max(2000, int(1.5 * len(orig)))
    if not (8 <= len(new) <= limit):
        return "length"
    if norm(new) == norm(orig):
        return "identical"
    if REFUSAL_RE.search(new) and not REFUSAL_RE.search(orig):
        return "refusal"
    if style == "with_content" and META_RE.search(new) and not META_RE.search(orig):
        return "meta"
    if style != "with_content" and len(new) > 3 * len(orig) + 300:
        return "too_long" 
    return None


def job_key(text, style):
    return hashlib.sha1(f"{style}||{text}".encode()).hexdigest()[:16]


def run_job(args, job):
    orig, cat, style = job["orig"], job["category"], job["style"]
    intent = INTENT.get(cat, "an ordinary request; keep exactly the same task")
    user = (f"Style: {STYLES[style]}\n"
            f"Intent to preserve: the user {intent}.\n\n"
            f"Prompt to rewrite:\n{orig}\n\n/no_think")
    for attempt in range(3):
        try:
            system = SYSTEM_CONTENT if style == "with_content" else SYSTEM
            new = clean(chat(args.base_url, args.model,
                             [{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                             temperature=0.9, max_tokens=args.max_tokens))
            why = valid(orig, new, style)
            if why:
                return {**job, "status": f"rejected:{why}", "prompt_text": new}
            if args.verify:
                ans = chat(args.base_url, args.model,
                           [{"role": "user", "content": VERIFY.format(a=orig, b=new)}],
                           temperature=0.0, max_tokens=args.max_tokens).lower()
                if not ans.startswith("yes"):
                    return {**job, "status": "rejected:intent", "prompt_text": new}
            return {**job, "status": "ok", "prompt_text": new}
        except Exception as e:  # server error, empty reply, network hiccup
            if attempt == 2:
                return {**job, "status": "error", "error": str(e)[:300], "prompt_text": ""}
            time.sleep(3 * (attempt + 1))


def build_jobs(train_csv, attack_n, benign_n, per_cat=None, styles=None, seed=SEED):
    df = pd.read_csv(train_csv)
    rng = random.Random(seed)
    style_names = styles or DEFAULT_STYLES

    if style_names == ["with_content"]:
        text = df["prompt_text"].astype(str)
        df = df[(text.str.len() <= CONTENT_MAX_ORIG_CHARS) & ~text.str.contains("\n\n")]
    jobs = []
    per_cat = per_cat or {}

    attacks = df[df["label"] == "distillation_attack"]
    for cat, g in attacks.groupby("attack_category"):
        n = per_cat.get(cat, attack_n)
        for i, (_, r) in enumerate(g.sample(n=min(n, len(g)), random_state=seed).iterrows()):
            jobs.append({"orig": r["prompt_text"], "label": "distillation_attack",
                         "category": cat, "subcategory": r.get("subcategory", ""),
                         "orig_source": r.get("source", "distillery"),
                         "style": style_names[i % len(style_names)]})

    benign = df[df["label"] == "benign"].sample(n=min(benign_n, (df["label"] == "benign").sum()),
                                                random_state=seed)
    for i, (_, r) in enumerate(benign.iterrows()):
        jobs.append({"orig": r["prompt_text"], "label": "benign",
                     "category": r["attack_category"], "subcategory": r.get("subcategory", ""),
                     "orig_source": r.get("source", ""),
                     "style": style_names[i % len(style_names)]})

    rng.shuffle(jobs)
    for j in jobs:
        j["key"] = job_key(j["orig"], j["style"])
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:1234/v1")
    p.add_argument("--model", help="model id as shown by --list-models")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--train-csv", default="data/processed/train.csv")
    p.add_argument("--out", default="data/augmented/aug.jsonl")
    p.add_argument("--attack-n", type=int, default=200, help="rewrites per attack category")
    p.add_argument("--attack-n-per", default="",
                   help="per-category override, e.g. "
                        "'reward_model_grading=220,safety_boundary_probe=220,capability_mapping=60'")
    p.add_argument("--benign-n", type=int, default=1200, help="benign rewrites in total")
    p.add_argument("--workers", type=int, default=2, help="parallel requests to LM Studio")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="per reply; thinking models need room to reason before answering")
    p.add_argument("--styles", default="",
                   help=f"comma list from {list(STYLES)}; default = the 5 v4 styles")
    p.add_argument("--seed", type=int, default=SEED,
                   help="sampling seed; use a new one to rewrite different originals than earlier runs")
    p.add_argument("--verify", action="store_true",
                   help="second LLM call to check the intent was preserved (doubles the time)")
    args = p.parse_args()

    if args.list_models:
        with urllib.request.urlopen(f"{args.base_url}/models", timeout=10) as r:
            for m in json.loads(r.read())["data"]:
                print(m["id"])
        return
    if not args.model:
        sys.exit("--model is required (see --list-models)")

    per_cat = {}
    
    for part in filter(None, args.attack_n_per.split(",")):
        cat, n = part.split("=")
        per_cat[cat.strip()] = int(n)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()] or DEFAULT_STYLES
    unknown = [s for s in styles if s not in STYLES]
    
    if unknown:
        sys.exit(f"unknown style(s): {unknown}; choose from {list(STYLES)}")
    
    jobs = build_jobs(args.train_csv, args.attack_n, args.benign_n, per_cat, styles, args.seed)
    
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()

    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["status"] == "ok" or (r["status"].startswith("rejected") and r.get("prompt_text")):
                        done.add(r["key"])
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(jobs):,} jobs, {len(done):,} already done, {len(todo):,} to go "
          f"(styles={styles}, workers={args.workers}, verify={args.verify})")

    stats, start = {}, time.time()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_job, args, j) for j in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            stats[res["status"]] = stats.get(res["status"], 0) + 1
            if res["status"] == "error" and stats["error"] <= 3:
                print(f"  ERROR: {res['error']}")
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  {i:,}/{len(todo):,}  {rate:.2f}/s  ETA {eta:.0f} min  {stats}")

    print("\nDone.", stats)


if __name__ == "__main__":
    main()
