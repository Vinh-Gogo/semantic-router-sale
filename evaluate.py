import json
import os
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore') # Ẩn các warning không cần thiết

from src.utils.read_jsonl import read_jsonl
from src.test.test import embeddings, predict_3_class

def run_auto_test():
    base_dir = os.path.dirname(__file__)
    test_file = os.path.join(base_dir, "data", "test_cases.jsonl")
    results_file = os.path.join(base_dir, "data", "test_results.jsonl") # File xuất kết quả
    
    if not os.path.exists(test_file):
        print("❌ Không tìm thấy file data/test_cases.jsonl")
        return

    # 1. LOAD VÀ NHÚNG DATASET
    print("🔄 Đang nạp dữ liệu từ Knowledge Base...")
    datasets = {
        "tao_lao": {"path": os.path.join(base_dir, "data/tao-lao.jsonl"), "data": [], "embeddings": None},
        "crawl_data": {"path": os.path.join(base_dir, "data/crawl-data.jsonl"), "data": [], "embeddings": None},
        "binh_thuong": {"path": os.path.join(base_dir, "data/binh-thuong.jsonl"), "data": [], "embeddings": None}
    }
    
    for name, ds in datasets.items():
        if os.path.exists(ds['path']):
            ds['data'] = read_jsonl(ds['path'], deduplicate=True)
            messages = [item['conversation'] for item in ds['data'] if item['conversation'].strip()]
            if messages:
                ds['embeddings'] = embeddings.embed_documents(messages)
    
    print("✅ Đã nạp xong Vector Database. Bắt đầu chạy Test Cases...\n")

    # 2. CHUẨN BỊ CHẤM ĐIỂM
    labels = ["binh_thuong", "crawl_data", "tao_lao"]
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    
    total_cases = 0
    correct_cases = 0
    total_cumulative = 0
    results_log = [] # Mảng lưu dữ liệu để xuất file

    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"{'#':<3} | {'KỲ VỌNG':<12} | {'AI DỰ ĐOÁN':<12} | {'CUMULATIVE':<10} | {'STATUS'}")
    print("-" * 65)

    # 3. CHẠY TỪNG TEST CASE
    for idx, line in enumerate(lines, 1):
        if not line.strip(): continue
        item = json.loads(line)
        
        expected_label = item["expected_label"]
        conversation = item["conversation"]
        
        chat_lines = conversation.split('\n')
        formatted_lines = [l.strip() for l in chat_lines if l.strip()]
        if not formatted_lines: continue
        
        full_text = "\n".join(formatted_lines)
        last_msg = formatted_lines[-1]
        
        # Tính toán vector similarity
        q_emb_full = np.array(embeddings.embed_query(full_text))
        n_full = np.linalg.norm(q_emb_full)
        q_norm_full = q_emb_full / n_full if n_full > 0 else q_emb_full
        
        if len(formatted_lines) > 1:
            q_emb_last = np.array(embeddings.embed_query(last_msg))
            n_last = np.linalg.norm(q_emb_last)
            q_norm_last = q_emb_last / n_last if n_last > 0 else q_emb_last
        else:
            q_norm_last = q_norm_full

        scores_similar = {"tao_lao": 0.0, "crawl_data": 0.0, "binh_thuong": 0.0}
        
        for name, ds in datasets.items():
            if ds['embeddings']:
                mat = np.array(ds['embeddings'])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                norm_mat = mat / norms
                
                sims_full = np.dot(norm_mat, q_norm_full)
                sims_last = np.dot(norm_mat, q_norm_last) if len(formatted_lines) > 1 else sims_full
                
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