from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

import sqlite3
import sqlite_vec
from langchain_core.documents import Document
from src.utils.vector_store import SQLiteVec
from src.utils.read_jsonl import read_jsonl
from src.test.test import (
    embeddings,
    predict_3_class
)


load_dotenv()
TOP_K_CONFIG = int(os.getenv("TOP_K"))

# --- SETUP SQLITE VEC ---
# 1. Khởi tạo kết nối và load extension sqlite-vec
db_conn = sqlite3.connect("chat_history.db", check_same_thread=False)
db_conn.enable_load_extension(True)
sqlite_vec.load(db_conn)
db_conn.enable_load_extension(False)

# 2. Lấy số chiều (dimension) của model embedding một cách linh động
# Nếu bạn dùng Qwen3-0.6B thì thường là 1024 hoặc 1536
sample_vec = embeddings.embed_query("test")
embed_dim = len(sample_vec)

# 3. Tạo bảng VIRTUAL chứa vector và text
# Sử dụng dấu + trước 'text' để định nghĩa auxiliary column lưu trữ metadata song song với vector
db_conn.execute(f'''
    CREATE VIRTUAL TABLE IF NOT EXISTS history_vectors 
    USING vec0(
        text_embedding float[{embed_dim}],
        +text TEXT
    );
''')
db_conn.commit()

# 4. Khởi tạo instance của class SQLiteVec từ vector_store.py
vector_store = SQLiteVec(table="history_vectors", connection=db_conn, embedding=embeddings)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        messages = [item['conversation'] for item in ds['data'] if item['conversation'].strip()]
        ds['embeddings'] = embeddings.embed_documents(messages)
        print(f"✅ {name}: {len(ds['embeddings'])} vectors")

    print("✅ Tất cả dataset đã sẵn sàng!")


@app.websocket("/ws/calculate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            # --- BẮT ĐẦU CẬP NHẬT: Xử lý chuỗi ---
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "messages" in payload:
                    lines = payload["messages"]
                else:
                    lines = raw.split('\n')
            except json.JSONDecodeError:
                lines = raw.split('\n')

            if not lines or all(not m.strip() for m in lines):
                await websocket.send_json({
                    "tao_lao": 0, "crawl_data": 0, "binh_thuong": 0,
                    "top": {"tao_lao": [], "crawl_data": [], "binh_thuong": []},
                    "detail": {}
                })
                continue

            # Format lại: Thêm "Customer: " vào đầu mỗi dòng
            formatted_lines = [f"Customer: {m.strip()}" for m in lines if m.strip()]
            full_conversation = "\n".join(formatted_lines)
            last_message = formatted_lines[-1] if formatted_lines else ""
            # --- KẾT THÚC CẬP NHẬT ---

            # 1. Dự đoán điểm tổng quát bằng Centroid (Trọng số 50% Last - 50% Full)
            if len(formatted_lines) > 1:
                _, scores_full = predict_3_class(full_conversation)
                _, scores_last = predict_3_class(last_message)
                
                scores = {
                    "TAO_LAO": 0.5 * scores_full.get("TAO_LAO", 0) + 0.5 * scores_last.get("TAO_LAO", 0),
                    "CRAWL_DATA": 0.5 * scores_full.get("CRAWL_DATA", 0) + 0.5 * scores_last.get("CRAWL_DATA", 0),
                    "BINH_THUONG": 0.5 * scores_full.get("BINH_THUONG", 0) + 0.5 * scores_last.get("BINH_THUONG", 0)
                }
            else:
                _, scores = predict_3_class(full_conversation)

            # 2. Tìm Top K câu giống nhất trong từng dataset
            top_results = {"tao_lao": [], "crawl_data": [], "binh_thuong": []}
            
            try:
                # Embed câu Full Conversation
                query_emb_full = np.array(embeddings.embed_query(full_conversation))
                norm_full = np.linalg.norm(query_emb_full)
                norm_query_full = query_emb_full / norm_full if norm_full > 0 else query_emb_full
                
                # Embed câu Last Message (nếu có nhiều hơn 1 câu)
                if len(formatted_lines) > 1:
                    query_emb_last = np.array(embeddings.embed_query(last_message))
                    norm_last = np.linalg.norm(query_emb_last)
                    norm_query_last = query_emb_last / norm_last if norm_last > 0 else query_emb_last
                else:
                    norm_query_last = norm_query_full

                for name, ds in datasets.items():
                    if ds['embeddings']:
                        mat = np.array(ds['embeddings'])
                        norms = np.linalg.norm(mat, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        norm_mat = mat / norms
                        
                        # Tính Cosine Similarity của Full
                        sims_full = np.dot(norm_mat, norm_query_full)
                        
                        # Tính Cosine Similarity của Last và Mix 50-50
                        if len(formatted_lines) > 1:
                            sims_last = np.dot(norm_mat, norm_query_last)
                            sims = 0.5 * sims_full + 0.5 * sims_last
                        else:
                            sims = sims_full
                        
                        # Lấy index của Top K
                        top_k = min(TOP_K_CONFIG, len(sims))
                        top_indices = np.argsort(sims)[-top_k:][::-1]
                        
                        for idx in top_indices:
                            text = ds['data'][idx]['conversation']
                            short_text = text[:20] + "..." if len(text) > 20 else text
                            top_results[name].append({
                                "text": short_text,
                                "score": round(float(sims[idx]) * 100, 1)
                            })
            except Exception as e:
                print(f"Lỗi khi tính Top 유사 (Similarity): {e}")

            # 3. Trả kết quả về cho Frontend
            result = {
                "tao_lao": scores.get("TAO_LAO", 0),
                "crawl_data": scores.get("CRAWL_DATA", 0),
                "binh_thuong": scores.get("BINH_THUONG", 0),
                "top": top_results,
                "detail": {}
            }

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
    # Tách các dòng, lấy dòng cuối và format toàn bộ
    lines = data.full_conversation.split('\n')
    
    # --- MỚI: Đếm số từ trước khi xử lý (bỏ qua nếu <= 3 từ) ---
    raw_text = " ".join([line.strip() for line in lines if line.strip()])
    if len(raw_text.split()) <= 3:
        return {"status": "skipped", "message": "Input quá ngắn (<= 3 từ), bỏ qua dự đoán."}
    # ------------------------------------------------------------

    formatted_lines = [f"Customer: {line.strip()}" for line in lines if line.strip()]
    formatted_full_conversation = "\n".join(formatted_lines)
    last_message = formatted_lines[-1] if formatted_lines else ""

    # Tính điểm 50% Full - 50% Last
    if len(formatted_lines) > 1:
        _, scores_full = predict_3_class(formatted_full_conversation)
        _, scores_last = predict_3_class(last_message)
        
        scores = {
            "TAO_LAO": 0.5 * scores_full.get("TAO_LAO", 0) + 0.5 * scores_last.get("TAO_LAO", 0),
            "CRAWL_DATA": 0.5 * scores_full.get("CRAWL_DATA", 0) + 0.5 * scores_last.get("CRAWL_DATA", 0),
            "BINH_THUONG": 0.5 * scores_full.get("BINH_THUONG", 0) + 0.5 * scores_last.get("BINH_THUONG", 0)
        }
        predicted_label = max(scores, key=scores.get) if scores else "UNKNOWN"
    else:
        predicted_label, scores = predict_3_class(formatted_full_conversation)

    # --- LƯU VÀO JSONL (GIỮ NGUYÊN) ---
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": "Customer: " + data.text,
        "full_conversation": formatted_full_conversation,
        "scores": {
            "TAO_LAO": scores.get("TAO_LAO", 0),
            "BINH_THUONG": scores.get("BINH_THUONG", 0),
            "CRAWL_DATA": scores.get("CRAWL_DATA", 0)
        },
        "predicted_label": predicted_label
    }
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # --- MỚI: LƯU VÀO SQLITE VEC ---
    try:
        # Bạn có thể lưu full_conversation hoặc kết hợp label vào text tùy mục đích search sau này
        doc = Document(
            page_content=f"[{predicted_label}] {formatted_full_conversation}", 
            metadata={"label": predicted_label} # Tạm thời metadata chưa lưu vào SQLite theo class của bạn, nhưng cứ gói vào cho chuẩn cấu trúc
        )
        vector_store.add_documents([doc])
        print(f"✅ Đã lưu vector thành công vào SQLiteVec: {predicted_label}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu vào SQLiteVec: {e}")

    return {"status": "ok", "logged": True}

# ==========================================
# CÁC API CHO TAB QUẢN LÝ DỮ LIỆU
# ==========================================

@app.get("/api/history")
async def get_history():
    """Lấy danh sách 50 log gần nhất từ file jsonl"""
    history_data = []
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        history_data.append(json.loads(line))
                    except:
                        pass
    # Trả về 50 dòng mới nhất (đảo ngược mảng)
    return history_data[::-1][:50]

class RelabelRequest(BaseModel):
    full_conversation: str
    new_label: str # Nhận: BINH_THUONG, TAO_LAO, CRAWL_DATA

@app.post("/api/relabel")
async def relabel_message(req: RelabelRequest):
    """
    Khi user bấm lưu nhãn mới:
    1. Ghi vào file {nhãn_mới}.jsonl
    2. Cập nhật thẳng vào SQLite Vector
    3. Push trực tiếp vào RAM (biến datasets) để sử dụng được ngay lập tức
    """
    label_map = {
        "BINH_THUONG": "binh_thuong",
        "TAO_LAO": "tao_lao",
        "CRAWL_DATA": "crawl_data"
    }
    
    ds_key = label_map.get(req.new_label)
    if not ds_key: 
        return {"status": "error", "error": "Nhãn không hợp lệ."}

    # 1. Cập nhật vào File JSONL tương ứng của nhãn đó
    target_ds = datasets[ds_key]
    new_entry = {
        "conversation": req.full_conversation,
        # Nếu data của bạn cần các key khác thì thêm ở đây
    }
    
    try:
        with open(target_ds['path'], 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        return {"status": "error", "error": f"Lỗi ghi file: {str(e)}"}

    # 2. Cập nhật vào SQLiteVec Database
    try:
        doc = Document(
            page_content=f"[{req.new_label}] {req.full_conversation}", 
            metadata={"label": req.new_label}
        )
        vector_store.add_documents([doc])
    except Exception as e:
        print(f"Lỗi SQLite trong lúc relabel: {e}")

    # 3. Cập nhật RAM (Embeddings In-Memory) để chat tiếp theo khôn ra liền
    try:
        new_emb = embeddings.embed_query(req.full_conversation)
        target_ds['embeddings'].append(new_emb)
        target_ds['data'].append(new_entry)
        print(f"✅ Đã huấn luyện thêm vào RAM & SQLite: {req.new_label}")
    except Exception as e:
        print(f"Lỗi lúc đưa vector vào RAM: {e}")

    return {"status": "ok"}

@app.get("/")
async def get_index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
