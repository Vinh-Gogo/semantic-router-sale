from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
spec = importlib.util.spec_from_file_location("read_jsonl", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "utils", "read-jsonl.py"))
read_jsonl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read_jsonl_module)
read_jsonl = read_jsonl_module.read_jsonl

from src.test.test import (
    embeddings,
    cosine_similarity
)

app = FastAPI(title="Multi Dataset Similarity Visualizer")

datasets = {
    "tao_lao": {
        "path": "data/tao-lao.jsonl",
        "data": [],
        "embeddings": None
    },
    "crawl_data": {
        "path": "data/crawl-data.jsonl",
        "data": [],
        "embeddings": None
    },
    "binh_thuong": {
        "path": "data/binh-thuong.jsonl",
        "data": [],
        "embeddings": None
    }
}


@app.on_event("startup")
async def preload_all_datasets():
    print("🔄 Đang precompute embeddings cho 3 dataset...")

    for name, ds in datasets.items():
        print(f"⚡ Đang xử lý {name}...")
        ds['data'] = read_jsonl(ds['path'], deduplicate=True)
        messages = [item['message'] for item in ds['data'] if item['message'].strip()]
        ds['embeddings'] = embeddings.embed_documents(messages)
        print(f"✅ {name}: {len(ds['embeddings'])} vectors")

    print("✅ Tất cả dataset đã sẵn sàng!")


@app.websocket("/ws/calculate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
                messages = payload.get("messages", [raw])
            except json.JSONDecodeError:
                messages = [raw]

            if not messages or all(not m.strip() for m in messages):
                await websocket.send_json({
                    "tao_lao": 0, "crawl_data": 0, "binh_thuong": 0,
                    "top": {"tao_lao": [], "crawl_data": [], "binh_thuong": []},
                    "detail": {}
                })
                continue

            embs = [embeddings.embed_query(m) for m in messages]

            result = {}
            top_matches = {}
            detail = {"messages": messages, "datasets": {}}

            for name, ds in datasets.items():
                cache_norms = ds['embeddings'] / np.linalg.norm(
                    ds['embeddings'], axis=1, keepdims=True
                )

                all_top = []
                per_msg_detail = []

                for idx, emb in enumerate(embs):
                    input_norm = emb / np.linalg.norm(emb)
                    similarities = np.dot(cache_norms, input_norm)

                    top_indices = np.argsort(similarities)[-2:][::-1]
                    top_scores = similarities[top_indices]

                    msg_score = round(float(np.mean(top_scores)) * 100, 2)

                    msg_top = [
                        {
                            "text": ds['data'][i]['message'],
                            "score": round(float(similarities[i]) * 100, 2)
                        }
                        for i in top_indices
                    ]

                    per_msg_detail.append({
                        "index": idx + 1,
                        "message": messages[idx],
                        "top": msg_top,
                        "score": msg_score
                    })

                    all_top.extend(msg_top)

                # Tính điểm phiên
                scores_list = [d["score"] for d in per_msg_detail]
                n = len(scores_list)

                if n >= 2:
                    avg_old = round(sum(scores_list[:-1]) / (n - 1), 2)
                    last = scores_list[-1]
                    final_score = round((avg_old + last) / 2, 2)
                else:
                    avg_old = None
                    last = scores_list[0]
                    final_score = last

                result[name] = final_score

                detail["datasets"][name] = {
                    "per_message": per_msg_detail,
                    "avg_old": avg_old,
                    "last_score": last,
                    "final_score": final_score
                }

                # Loại trùng, lấy top 2
                seen = set()
                unique_top = []
                for item in sorted(all_top, key=lambda x: x['score'], reverse=True):
                    if item['text'] not in seen:
                        seen.add(item['text'])
                        unique_top.append(item)
                    if len(unique_top) >= 2:
                        break
                top_matches[name] = unique_top

            result['top'] = top_matches
            result['detail'] = detail
            await websocket.send_json(result)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


class ChatSubmit(BaseModel):
    text: str
    full_conversation: str

@app.post("/submit")
async def submit_chat_message(data: ChatSubmit):
    input_emb = embeddings.embed_query(data.full_conversation)

    scores = {}
    for name, ds in datasets.items():
        input_norm = input_emb / np.linalg.norm(input_emb)
        cache_norms = ds['embeddings'] / np.linalg.norm(ds['embeddings'], axis=1, keepdims=True)
        similarities = np.dot(cache_norms, input_norm)
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_scores = similarities[top_indices]
        scores[name] = round(float(np.mean(top_scores)) * 100, 2)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": data.text,
        "full_conversation": data.full_conversation,
        "scores": scores,
        "predicted_label": max(scores, key=scores.get)
    }

    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {"status": "ok", "logged": True}

@app.get("/")
async def get_index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
