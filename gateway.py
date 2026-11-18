
import asyncio
import datetime as dt
import hashlib
import os
import time
import uuid

import httpx
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import laya
from make_rows import QID, QUESTION

UPSTREAM = os.getenv("UPSTREAM", "http://localhost:1234/v1") 
ES_URL = os.getenv("ES_URL", "https://localhost:9200")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_CA = os.getenv("ES_CA")
MODELS = os.getenv(
    "DG_MODELS",
    "models/distilguard-v4,models/distilguard-v4s2,models/distilguard-v4s3",
).split(",")


THR = float(os.getenv("DG_THR", "0.485"))
PROMPT_MODE = os.getenv("DG_PROMPT_MODE", "inline")
if PROMPT_MODE not in ("inline", "split"):
    raise SystemExit(f"DG_PROMPT_MODE must be 'inline' or 'split', got {PROMPT_MODE!r}")
BATCH, MAX_WAIT_S = 32, 0.2 

QUESTIONS = {QID: QUESTION}
app = FastAPI(title="DistilGuard gateway")
agents = [laya.load(m) for m in MODELS]
es = AsyncElasticsearch(ES_URL, api_key=ES_API_KEY, ca_certs=ES_CA)
queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
http = httpx.AsyncClient(timeout=300)


def sha(text: str, n: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def last_user_text(body: dict) -> str:
    """Newest user message: the part of the request that is new this turn."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, list): 
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c or "").strip()
    return ""


def score_batch(texts: list[str]) -> list[list[float]]:
    """Per-seed scores, shape [len(texts)][len(agents)]."""
    states = [{"prompt": t} for t in texts]
    per_model = []
    for a in agents:
        res = a.predict_batch(states, QUESTIONS, batch_size=BATCH, sort_by_length=True)
        per_model.append([r["answers"][QID]["noul"] for r in res])
    return [list(x) for x in zip(*per_model)]


async def scorer():
    loop = asyncio.get_running_loop()
    while True:
        items = [await queue.get()]
        deadline = loop.time() + MAX_WAIT_S
        while len(items) < BATCH and (left := deadline - loop.time()) > 0:
            try:
                items.append(await asyncio.wait_for(queue.get(), left))
            except asyncio.TimeoutError:
                break
        try:
            seeds = await asyncio.to_thread(score_batch, [i["prompt"] for i in items])
        except Exception as e: 
            print(f"scoring failed for {len(items)} events: {e}")
            continue

        actions = []
        for item, s in zip(items, seeds):
            mean = sum(s) / len(s)
            if PROMPT_MODE == "inline":
                event = dict(item)
            else:
                event = {k: v for k, v in item.items() if k != "prompt"}
            event["distilguard"] = {
                "score": round(mean, 4),
                "seeds": [round(x, 4) for x in s],
                "flag": mean >= THR,
                "threshold": THR,
            }
            actions.append({"_op_type": "create", "_index": "llm-events", "_source": event})
            if PROMPT_MODE == "split":
                day = item["@timestamp"][:10].replace("-", ".")
                actions.append({
                    "_op_type": "create",
                    "_index": f"llm-prompts-raw-{day}",
                    "_source": {"@timestamp": item["@timestamp"], "event_id": item["event_id"],
                                "prompt": item["prompt"]},
                })
        try:
            ok, errors = await async_bulk(es, actions, raise_on_error=False)
            if errors:
                print(f"indexing: {ok} ok, {len(errors)} errors, first: {errors[0]}")
        except Exception as e:
            print(f"indexing failed for {len(items)} events: {e}")


@app.on_event("startup")
async def start():
    app.state.scorer = asyncio.create_task(scorer())


@app.post("/v1/chat/completions")
async def chat(request: Request):

    auth = request.headers.get("authorization", "")

    if not auth.lower().startswith("bearer ") or len(auth) < 12:
        raise HTTPException(401, "missing API key")
    client_key = auth[7:].strip()

    body = await request.json()
    body["stream"] = False

    start = time.perf_counter()
    r = await http.post(f"{UPSTREAM}/chat/completions", json=body)
    latency_ms = (time.perf_counter() - start) * 1000

    prompt = last_user_text(body)

    if prompt:

        event = {
            "@timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "api_key_id": sha(client_key, 12),  # never store the key itself
            "session_id": request.headers.get("x-session-id", ""),
            "client_ip": request.client.host if request.client else None,
            "model": body.get("model", ""),
            "turns": len(body.get("messages", [])),
            "prompt_sha256": sha(prompt),
            "prompt_chars": len(prompt),
            "upstream_status": r.status_code,
            "upstream_latency_ms": round(latency_ms, 1),
            "prompt": prompt,
        }
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            print("score queue full, dropping event")  

    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except ValueError:
        return JSONResponse({"error": r.text[:500]}, status_code=502)
