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

# Cache embeddings cho 3 dataset
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
    },
    # "hello": {
    #     "path": "data/hello.jsonl",
    #     "data": [],
    #     "embeddings": None
    # }
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


def calculate_dataset_score(input_embedding, dataset_embeddings):
    """Tính điểm trung bình similarity cho 1 dataset"""
    input_norm = input_embedding / np.linalg.norm(input_embedding)
    cache_norms = dataset_embeddings / np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)
    similarities = np.dot(cache_norms, input_norm)
    return float(np.mean(similarities)) * 100


@app.websocket("/ws/calculate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            
            if not text or len(text.strip().split()) < 1:
                await websocket.send_json({
                    "tao_lao": 0,
                    "crawl_data": 0,
                    "binh_thuong": 0,
                    # "hello": 0,s
                })
                continue
            
            # Tạo embedding cho input 1 lần duy nhất
            input_emb = embeddings.embed_query(text)
            
            # Tính score đồng thời cho cả 3 dataset
            result = {}
            top_matches = {}
            
            for name, ds in datasets.items():
                input_norm = input_emb / np.linalg.norm(input_emb)
                cache_norms = ds['embeddings'] / np.linalg.norm(ds['embeddings'], axis=1, keepdims=True)
                similarities = np.dot(cache_norms, input_norm)
                
                # Lấy top 5 gần nhất
                top_indices = np.argsort(similarities)[-2:][::-1]
                top_scores = similarities[top_indices]
                
                # Tính điểm trung bình CHỈ từ top 5, không dùng toàn bộ dataset
                result[name] = round(float(np.mean(top_scores)) * 100, 2)
                
                top_matches[name] = [
                    {
                        "text": ds['data'][i]['message'],
                        "score": round(float(similarities[i]) * 100, 2)
                    }
                    for i in top_indices
                ]
            
            result['top'] = top_matches
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
    """Endpoint nhận tin nhắn chat và tự động log vào file lưu trữ"""
    
    # Tính score cho tin nhắn này
    input_emb = embeddings.embed_query(data.full_conversation)
    
    scores = {}
    for name, ds in datasets.items():
        input_norm = input_emb / np.linalg.norm(input_emb)
        cache_norms = ds['embeddings'] / np.linalg.norm(ds['embeddings'], axis=1, keepdims=True)
        similarities = np.dot(cache_norms, input_norm)
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_scores = similarities[top_indices]
        scores[name] = round(float(np.mean(top_scores)) * 100, 2)
    
    # Ghi log vào file vĩnh viễn
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
