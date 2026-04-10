import time
import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.utils.read_jsonl import read_jsonl

import os
import json
import numpy as np
import requests
    
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",
    base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    # base_url="https://api.novita.ai/openai",
    # api_key=SecretStr(os.getenv("NOVITA_API_KEY", "text")),
    check_embedding_ctx_length=False,
    chunk_size=32
)

def embed_with_retry(texts):
    """Hàm hỗ trợ nhúng danh sách văn bản với cơ chế đợi nếu gặp giới hạn rate limit"""
    try:
        # Nhờ chunk_size=32 đã cài ở turn trước, 
        # hàm này sẽ tự động chia nhỏ texts thành các batch 32.
        return embeddings.embed_documents(texts)
    except Exception as e:
        if "429" in str(e):
            print("⏳ Chạm giới hạn Rate Limit. Đang nghỉ 30s...")
            time.sleep(3)
            return embeddings.embed_documents(texts)
        raise e

def run_auto_test():
    base_dir = os.path.dirname(__file__)
    test_file = os.path.join(base_dir, "data", "test_cases.jsonl")
    results_file = os.path.join(base_dir, "data", "test_results.jsonl")
    
    if not os.path.exists(test_file): return

    # 1. LOAD DATASET (Giữ nguyên logic cũ nhưng dùng embed_with_retry)
    print("🔄 Đang nạp dữ liệu từ Knowledge Base...")
    datasets = {
        "tao_lao": {"path": os.path.join(base_dir, "data/tao-lao.jsonl"), "embeddings": None},
        "crawl_data": {"path": os.path.join(base_dir, "data/crawl-data.jsonl"), "embeddings": None},
        "binh_thuong": {"path": os.path.join(base_dir, "data/binh-thuong.jsonl"), "embeddings": None}
    }
    
    for name, ds in datasets.items():
        if os.path.exists(ds['path']):
            data = read_jsonl(ds['path'], deduplicate=True)
            msgs = [item['conversation'] for item in data if item['conversation'].strip()]
            if msgs:
                ds['embeddings'] = embed_with_retry(msgs)

    # 2. GOM TOÀN BỘ TEST CASES ĐỂ NHÚNG 1 LẦN (BATCHING)
    print("📦 Đang gom batch Test Cases để tối ưu API...")
    test_items = []
    all_texts_to_embed = []
    
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                chat_lines = [l.strip() for l in item["conversation"].split('\n') if l.strip()]
                if not chat_lines: continue
                
                full_text = "\n".join(chat_lines)
                last_msg = chat_lines[-1]
                
                # Lưu index để tí nữa map lại
                item['_full_idx'] = len(all_texts_to_embed)
                all_texts_to_embed.append(full_text)
                
                item['_last_idx'] = len(all_texts_to_embed)
                all_texts_to_embed.append(last_msg)
                
                item['_chat_lines'] = chat_lines
                test_items.append(item)

    # Nhúng toàn bộ 1 lần (API sẽ chỉ tốn (tổng_văn_vản / 32) requests)
    print(f"📡 Đang gửi {len(all_texts_to_embed)} chuỗi văn bản lên Server...")
    all_vectors = np.array(embed_with_retry(all_texts_to_embed))
    
    # Chuẩn hóa toàn bộ vector nhúng
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors_norm = all_vectors / np.where(norms == 0, 1, norms)

    # 3. CHẠY CHẤM ĐIỂM (Không gọi API nữa, chỉ tính toán CPU)
    print("✅ Bắt đầu so khớp vector...")
    labels = ["binh_thuong", "crawl_data", "tao_lao"]
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    
    # ---> BỔ SUNG KHAI BÁO BIẾN Ở ĐÂY <---
    total_cases = 0
    correct_cases = 0
    total_cumulative = 0
    results_log = []
    # ------------------------------------

    for idx, item in enumerate(test_items, 1):
        expected_label = item["expected_label"]
        conversation = item["conversation"]
        
        q_norm_full = all_vectors_norm[item['_full_idx']]
        q_norm_last = all_vectors_norm[item['_last_idx']]
        
        scores_similar = {name: 0.0 for name in datasets.keys()}
        for name, ds in datasets.items():
            if ds['embeddings'] is not None:
                mat = np.array(ds['embeddings'])
                # Chuẩn hóa database vector
                db_norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norm_mat = mat / np.where(db_norms == 0, 1, db_norms)
                
                sims_full = np.dot(norm_mat, q_norm_full)
                sims_last = np.dot(norm_mat, q_norm_last)
                
                sims = 0.5 * sims_full + 0.5 * sims_last
                top_k = min(2, len(sims))
                if top_k > 0:
                    top_scores = np.sort(sims)[-top_k:]
                    scores_similar[name] = round(float(np.mean(top_scores)) * 100, 2)
                    
        # Chốt dự đoán
        predicted_label = max(scores_similar, key=scores_similar.get)
        
        # Tính Probability và Cumulative
        total_score = sum(scores_similar.values())
        cumulative = 0
        probs_dict = {"tao_lao": 0.0, "crawl_data": 0.0, "binh_thuong": 0.0}
        if total_score > 0:
            probs_dict = {k: round((v / total_score) * 100, 2) for k, v in scores_similar.items()}
            sorted_probs = sorted(probs_dict.values(), reverse=True)
            cumulative = round(sorted_probs[0] + sorted_probs[1], 2)
            
        # Đánh giá đúng/sai
        is_correct = (predicted_label == expected_label)
        status_icon = "✅ Đúng" if is_correct else "❌ SAI"
        
        if expected_label in labels and predicted_label in labels:
            matrix[expected_label][predicted_label] += 1
            total_cases += 1
            total_cumulative += cumulative
            if is_correct:
                correct_cases += 1
                
        # Ghi nhận vào mảng log để xuất file
        result_entry = {
            "test_id": idx,
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "best_label": scores_similar,
            "probabilities": probs_dict,
            "cumulative_top2_prob": cumulative,
            "conversation": conversation
        }
        results_log.append(result_entry)

        # In ra console
        print(f"{idx:<3} | {expected_label:<12} | {predicted_label:<12} | {cumulative:>5.1f}%     | {status_icon}")

    # --- TIẾN HÀNH GHI RA FILE JSONL ---
    with open(results_file, "w", encoding="utf-8") as f:
        for entry in results_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 4. IN BÁO CÁO TỔNG QUAN
    print("\n" + "="*50)
    print("🚀 BÁO CÁO KẾT QUẢ AUTOMATED TEST")
    print("="*50)
    if total_cases > 0:
        accuracy = (correct_cases / total_cases) * 100
        avg_cumul = total_cumulative / total_cases
        print(f"🎯 ACCURACY (Tỷ lệ đúng)    : {accuracy:.2f}% ({correct_cases}/{total_cases})")
        print(f"🌟 AVG CUMULATIVE TOP-2     : {avg_cumul:.2f}%")
        print(f"💾 File báo cáo chi tiết    : {results_file}")
        print("-" * 50)
        print("🧩 MA TRẬN NHẦM LẪN (Dựa trên Test Cases):")
        print(f"{'Thực tế \ Dự đoán':<18} | {'BT':<4} | {'CD':<4} | {'TL':<4} |")
        print("-" * 43)
        short_labels = {"binh_thuong": "BT", "crawl_data": "CD", "tao_lao": "TL"}
        for actual in labels:
            row = f"Thực tế {short_labels[actual]:<9} | "
            for pred in labels:
                val = matrix[actual][pred]
                val_str = f"[{val}]" if actual == pred else str(val)
                row += f"{val_str:<4} | "
            print(row)
    else:
        print("Chưa có Test Case nào được thực thi hợp lệ.")

if __name__ == "__main__":
    run_auto_test()