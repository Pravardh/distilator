
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gateway", default="http://localhost:8000")
    p.add_argument("--key", required=True, help="client API key to send as (any string, 6+ chars)")
    p.add_argument("--file", required=True, help="one prompt per line")
    p.add_argument("--model", default="gemma-3-12b-it")
    p.add_argument("--sessions", type=int, default=1, help="spread prompts over N session ids")
    p.add_argument("--max-tokens", type=int, default=1, help="keep tiny: detection only needs the prompt")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    args = p.parse_args()

    with open(args.file, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    sessions = [str(uuid.uuid4()) for _ in range(max(1, args.sessions))]
    key_id = hashlib.sha256(args.key.encode("utf-8")).hexdigest()[:12]
    print(f"key '{args.key}' -> api_key_id {key_id} in Elastic, {len(prompts)} prompts\n")

    fails = 0
    for i, text in enumerate(prompts, 1):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": args.max_tokens,
        }
        req = urllib.request.Request(
            f"{args.gateway}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.key}",
                "X-Session-Id": sessions[(i - 1) % len(sessions)],
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:
            status = f"error: {e}"
        print(f"{i:>3}/{len(prompts)}  {status}  {text[:70]}")
        fails = fails + 1 if str(status).startswith("error") else 0
        if fails >= 3:
            raise SystemExit(f"\n3 connection errors in a row: is the gateway running at {args.gateway}?")
        if args.delay:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
