from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
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
TOP_K_CONFIG = int(os.getenv("TOP_K", 2))

# --- SETUP SQLITE VEC ---
# 1. Khởi tạo kết nối và load extension sqlite-vec
db_conn = sqlite3.connect("chat_history.db", check_same_thread=False)
db_conn.enable_load_extension(True)
sqlite_vec.load(db_conn)
db_conn.enable_load_extension(False)

# 2. Lấy số chiều (dimension) của model embedding một cách linh động
sample_vec = embeddings.embed_query("test")
embed_dim = len(sample_vec)

# 3. Tạo bảng VIRTUAL chứa vector và text
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
        if messages:
            ds['embeddings'] = embeddings.embed_documents(messages)
            print(f"✅ {name}: {len(ds['embeddings'])} vectors")
        else:
            ds['embeddings'] = []
            print(f"⚠️ {name}: 0 vectors (Dataset rỗng)")

    print("✅ Tất cả dataset đã sẵn sàng!")


@app.websocket("/ws/calculate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            # --- Xử lý chuỗi ---
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

            # 1. Dự đoán điểm tổng quát bằng Centroid (Trọng số 50% Last - 50% Full)
            if len(formatted_lines) > 1:
                _, scores_full = predict_3_class(full_conversation)
                _, scores_last = predict_3_class(last_message)
                
                scores_tao_lao = 0.5 * scores_full.get("TAO_LAO", 0) + 0.5 * scores_last.get("TAO_LAO", 0)
                scores_crawl_data = 0.5 * scores_full.get("CRAWL_DATA", 0) + 0.5 * scores_last.get("CRAWL_DATA", 0)
                scores_binh_thuong = 0.5 * scores_full.get("BINH_THUONG", 0) + 0.5 * scores_last.get("BINH_THUONG", 0)
            else:
                _, scores_raw = predict_3_class(full_conversation)
                scores_tao_lao = scores_raw.get("TAO_LAO", 0)
                scores_crawl_data = scores_raw.get("CRAWL_DATA", 0)
                scores_binh_thuong = scores_raw.get("BINH_THUONG", 0)

            # 2. Tìm Top K câu giống nhất
            top_results = {"tao_lao": [], "crawl_data": [], "binh_thuong": []}
            
            try:
                # Embed câu Full Conversation
                query_emb_full = np.array(embeddings.embed_query(full_conversation))
                norm_full = np.linalg.norm(query_emb_full)
                norm_query_full = query_emb_full / norm_full if norm_full > 0 else query_emb_full
                
                # Embed câu Last Message
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
                        
                        # Tính Cosine Similarity
                        sims_full = np.dot(norm_mat, norm_query_full)
                        
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
                            short_text = text[:80] + "..." if len(text) > 80 else text
                            top_results[name].append({
                                "text": short_text,
                                "score": round(float(sims[idx]) * 100, 1)
                            })
            except Exception as e:
                print(f"Lỗi khi tính Top Similarity: {e}")

            # 3. Trả kết quả về cho Frontend (Chuẩn hóa lowercase)
            result = {
                "tao_lao": scores_tao_lao,
                "crawl_data": scores_crawl_data,
                "binh_thuong": scores_binh_thuong,
                "top": top_results,
                "detail": {}
            }

            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("💡 Client đã ngắt kết nối WebSocket (Reload/Đóng tab).")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.close()
        except RuntimeError:
            pass

# 1. Thêm Optional vào import để handle id có thể null
from typing import Optional 

class ChatSubmit(BaseModel):
    id: Optional[int] = None # Thêm trường ID
    text: str
    full_conversation: str

@app.post("/submit")
async def submit_chat_message(data: ChatSubmit):
    lines = data.full_conversation.split('\n')
    raw_text = " ".join([line.strip() for line in lines if line.strip()])
    
    if len(raw_text.split()) <= 3:
        return {"status": "skipped", "message": "Input quá ngắn."}

    formatted_lines = [f"Customer: {line.strip()}" for line in lines if line.strip()]
    print(formatted_lines)
    formatted_full_conversation = "\n".join(formatted_lines)
    last_message = formatted_lines[-1] if formatted_lines else ""

    # ==========================================
    # 1. TÍNH ĐIỂM CENTROID (scores) - Trọng số 50/50
    # ==========================================
    if len(formatted_lines) > 1:
        _, c_scores_full = predict_3_class(formatted_full_conversation)
        _, c_scores_last = predict_3_class(last_message)
        scores_centroid = {
            "tao_lao": round(0.5 * c_scores_full.get("TAO_LAO", 0) + 0.5 * c_scores_last.get("TAO_LAO", 0), 2),
            "crawl_data": round(0.5 * c_scores_full.get("CRAWL_DATA", 0) + 0.5 * c_scores_last.get("CRAWL_DATA", 0), 2),
            "binh_thuong": round(0.5 * c_scores_full.get("BINH_THUONG", 0) + 0.5 * c_scores_last.get("BINH_THUONG", 0), 2)
        }
    else:
        _, c_scores_raw = predict_3_class(formatted_full_conversation)
        scores_centroid = {
            "tao_lao": round(c_scores_raw.get("TAO_LAO", 0), 2),
            "crawl_data": round(c_scores_raw.get("CRAWL_DATA", 0), 2),
            "binh_thuong": round(c_scores_raw.get("BINH_THUONG", 0), 2)
        }

    # ==========================================
    # 2. TÍNH ĐIỂM SIMILARITY (Trung bình Top-2)
    # ==========================================
    query_emb_full = np.array(embeddings.embed_query(formatted_full_conversation))
    norm_full = np.linalg.norm(query_emb_full)
    norm_query_full = query_emb_full / norm_full if norm_full > 0 else query_emb_full
    
    if len(formatted_lines) > 1:
        query_emb_last = np.array(embeddings.embed_query(last_message))
        norm_last = np.linalg.norm(query_emb_last)
        norm_query_last = query_emb_last / norm_last if norm_last > 0 else query_emb_last
    else:
        norm_query_last = norm_query_full

    scores_similar = {"tao_lao": 0.0, "crawl_data": 0.0, "binh_thuong": 0.0}

    try:
        for name, ds in datasets.items():
            if ds['embeddings']:
                mat = np.array(ds['embeddings'])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                norm_mat = mat / norms
                
                sims_full = np.dot(norm_mat, norm_query_full)
                
                if len(formatted_lines) > 1:
                    sims_last = np.dot(norm_mat, norm_query_last)
                    sims = 0.5 * sims_full + 0.5 * sims_last
                else:
                    sims = sims_full
                
                # Lấy Top 2 và tính trung bình
                top_k = min(2, len(sims))
                if top_k > 0:
                    top_scores = np.sort(sims)[-top_k:]
                    avg_score = float(np.mean(top_scores)) * 100
                    scores_similar[name] = round(avg_score, 2)
    except Exception as e:
        print(f"❌ Lỗi khi tính điểm Similarity: {e}")

    # Chốt dự đoán dựa trên điểm Similarity
    predicted_label = max(scores_similar, key=scores_similar.get) if any(scores_similar.values()) else "unknown"

    # ==========================================
    # 3. TÍNH XÁC SUẤT TÍCH LŨY (Cumulative Probability)
    # ==========================================
    cumulative_top2 = 0.0
    probs_dict = {"tao_lao": 0.0, "crawl_data": 0.0, "binh_thuong": 0.0}
    
    total_sim_score = sum(scores_similar.values())
    if total_sim_score > 0:
        # Chuẩn hóa về thang 100% (Probability Distribution)
        probs_dict = {k: round((v / total_sim_score) * 100, 2) for k, v in scores_similar.items()}
        # Lấy 2 xác suất cao nhất cộng lại
        sorted_probs = sorted(probs_dict.values(), reverse=True)
        if len(sorted_probs) >= 2:
            cumulative_top2 = round(sorted_probs[0] + sorted_probs[1], 2)

    # ==========================================
    # 4. GHI LOG & DATABASE
    # ==========================================
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    
    target_id = data.id
    all_entries = []
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    all_entries.append(json.loads(line))
                except Exception:
                    pass

    # Tạo JSON Entry với cấu trúc siêu đầy đủ
    new_entry = {
        "id": target_id if target_id else (max([int(e.get('id') or 0) for e in all_entries] + [0]) + 1),
        "predicted_label": predicted_label,
        "scores": scores_centroid,            # 1. Điểm Centroid 50/50
        "scores_similar": scores_similar,     # 2. Điểm Similarity Top-2
        "probabilities": probs_dict,          # 3. Xác suất phân bổ (%)
        "cumulative_top2_prob": cumulative_top2, # 4. Xác suất tích lũy Top 2
        "is_reviewed": False,                 
        "full_conversation": formatted_full_conversation,
        "timestamp": datetime.utcnow().isoformat()
    }

    if target_id:
        found = False
        for i, entry in enumerate(all_entries):
            if entry.get("id") == target_id:
                all_entries[i] = new_entry
                found = True
                break
        if not found: all_entries.append(new_entry)
    else:
        all_entries.append(new_entry)

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    try:
        if target_id is not None:
            db_conn.execute("DELETE FROM history_vectors WHERE text LIKE ?", (f"%ID:{target_id} %",)) 
            
        doc = Document(
            page_content=f"[{predicted_label}] ID:{new_entry['id']} {formatted_full_conversation}", 
            metadata={"label": predicted_label, "id": new_entry['id']}
        )
        vector_store.add_documents([doc])
        db_conn.commit()
    except Exception as e:
        print(f"❌ SQLite Error: {e}")

    return {"status": "ok", "id": new_entry['id']}

# ==========================================
# CÁC API CHO TAB QUẢN LÝ DỮ LIỆU
# ==========================================

@app.get("/api/history")
async def get_history():
    """Lấy danh sách log gần nhất, tự động gộp các đoạn chat bị lặp và ẨN các đoạn đã đánh giá"""
    raw_history = []
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # --- MỚI: Chỉ lấy những câu chưa được review (False hoặc chưa có field này) ---
                        if not item.get("is_reviewed", False):
                            raw_history.append(item)
                    except:
                        pass
                        
    # Đảo ngược mảng để xét từ tin nhắn mới nhất (hoàn chỉnh nhất) trở về trước
    raw_history = raw_history[::-1]
    
    filtered_history = []
    for item in raw_history:
        current_text = item.get("full_conversation", "")
        
        is_subset = False
        for kept_item in filtered_history:
            kept_text = kept_item.get("full_conversation", "")
            
            # Nếu đoạn text hiện tại trùng khớp hoàn toàn 
            # HOẶC là phần đầu của một đoạn text khác đã được giữ lại (cách nhau bởi \n)
            if kept_text == current_text or kept_text.startswith(current_text + "\n"):
                is_subset = True
                break
                
        # Nếu không bị trùng lặp/nằm trong đoạn dài hơn, thì đưa vào danh sách hiển thị
        if not is_subset:
            filtered_history.append(item)
            
    # Trả về tối đa 50 đoạn hội thoại đã được lọc sạch và CHƯA ĐÁNH GIÁ
    return filtered_history[:50]

class RelabelRequest(BaseModel):
    full_conversation: str
    new_label: str # Nhận trực tiếp: binh_thuong, tao_lao, crawl_data

@app.post("/api/relabel")
async def relabel_message(req: RelabelRequest):
    """
    Cập nhật toàn diện khi người dùng thay đổi nhãn:
    1. Sửa 'predicted_label' trong user_submitted_messages.jsonl (Để UI hiển thị đúng)
    2. Xóa dữ liệu khỏi các file nhãn sai (data/tao-lao.jsonl, v.v.)
    3. Thêm dữ liệu vào file nhãn đúng (data/binh-thuong.jsonl, v.v.)
    4. Đồng bộ lại SQLite Vector Store
    """
    target_label = req.new_label # Ví dụ: 'binh_thuong'
    conv_text = req.full_conversation

    # --- 1. CẬP NHẬT FILE LOG TỔNG (user_submitted_messages.jsonl) ---
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    if os.path.exists(log_file):
        updated_logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    # Tìm đúng đoạn hội thoại để đổi nhãn
                    if entry.get("full_conversation") == conv_text:
                        entry["predicted_label"] = target_label
                        entry["is_reviewed"] = True # <--- THÊM DÒNG NÀY: Đánh dấu đã duyệt xong
                    updated_logs.append(entry)
                except: continue
        
        # Ghi đè lại file log tổng với nhãn đã sửa
        with open(log_file, "w", encoding="utf-8") as f:
            for entry in updated_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- 2. XÓA KHỎI CÁC NHÃN SAI (Dọn dẹp các file dataset lẻ) ---
    for name, ds in datasets.items():
        if name != target_label:
            # Tìm và loại bỏ câu này khỏi RAM và File của các nhãn không liên quan
            original_data = ds['data']
            new_data = [item for item in original_data if item.get('conversation') != conv_text]
            
            if len(new_data) < len(original_data):
                # Nếu có sự thay đổi (đã tìm thấy và xóa), cập nhật lại RAM và ghi đè File
                ds['data'] = new_data
                # Cập nhật lại embeddings trong RAM (nếu có)
                if ds['embeddings']:
                    ds['embeddings'] = embeddings.embed_documents([x['conversation'] for x in new_data])
                
                with open(ds['path'], "w", encoding="utf-8") as f:
                    for item in new_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --- 3. THÊM VÀO NHÃN ĐÚNG (Dataset lẻ mục tiêu) ---
    target_ds = datasets[target_label]
    exists_in_target = any(item.get('conversation') == conv_text for item in target_ds['data'])
    
    if not exists_in_target:
        # Tạo ID mới tự tăng cho dataset này
        new_id = 1
        if target_ds['data']:
            try:
                new_id = max([int(it.get('id', 0)) for it in target_ds['data']]) + 1
            except:
                new_id = len(target_ds['data']) + 1

        new_entry = {"id": new_id, "conversation": conv_text}
        
        # Cập nhật RAM
        target_ds['data'].append(new_entry)
        new_emb = embeddings.embed_query(conv_text)
        if target_ds['embeddings'] is not None:
            target_ds['embeddings'].append(new_emb)
        else:
            target_ds['embeddings'] = [new_emb]

        # Ghi nối vào file nhãn đúng
        with open(target_ds['path'], "a", encoding="utf-8") as f:
            f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    # --- 4. ĐỒNG BỘ SQLITE VECTOR (Xóa cũ - Chèn mới) ---
    try:
        # Xóa vector cũ dựa trên nội dung (text)
        db_conn.execute("DELETE FROM history_vectors WHERE text LIKE ?", (f"%{conv_text}",))
        
        # Chèn vector mới với nhãn đã được đánh giá lại
        # Format: [nhãn_mới] ID:x Nội dung...
        doc = Document(
            page_content=f"[{target_label}] {conv_text}", 
            metadata={"label": target_label}
        )
        vector_store.add_documents([doc])
        db_conn.commit()
    except Exception as e:
        print(f"❌ Lỗi SQLite trong relabel: {e}")

    return {"status": "ok"}

@app.get("/api/reviewed_history")
async def get_reviewed_history():
    """Lấy danh sách log ĐÃ ĐÁNH GIÁ từ file jsonl"""
    raw_history = []
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_submitted_messages.jsonl")
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # CHỈ LẤY CÁC DÒNG ĐÃ REVIEW
                        if item.get("is_reviewed", False):
                            raw_history.append(item)
                    except:
                        pass
                        
    # Đảo ngược mảng để xét từ tin nhắn mới nhất
    raw_history = raw_history[::-1]
    
    filtered_history = []
    for item in raw_history:
        current_text = item.get("full_conversation", "")
        
        is_subset = False
        for kept_item in filtered_history:
            kept_text = kept_item.get("full_conversation", "")
            if kept_text == current_text or kept_text.startswith(current_text + "\n"):
                is_subset = True
                break
                
        if not is_subset:
            filtered_history.append(item)
            
    # Trả về tối đa 50 đoạn hội thoại ĐÃ KIỂM TRA
    return filtered_history[:50]

@app.get("/")
async def get_index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())