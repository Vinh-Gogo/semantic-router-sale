import os
import json
import numpy as np

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv()

# base_url from env
base_url = os.getenv("OPENAI_BASE_URL_EMBED")
try:
    resp = requests.get(f"{base_url}/models", timeout=5)
    model_id = resp.json()["data"][0]["id"]
    print(f"🔍 Auto-detected model: {model_id}")
except Exception as e:
    model_id = os.getenv("OPENAI_API_MODEL_NAME_EMBED")
    print(f"⚠️ Using fallback model: {model_id}")

embeddings = OpenAIEmbeddings(
    model=model_id,
    base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    check_embedding_ctx_length=False,
    # tiktoken_enabled=False,
)

# Cache embedding dataset (chỉ khởi tạo 1 lần)
_cached_embeddings = None
_cached_file_mtime = None


def read_tao_lao_data(file_path: str = "data/tao-lao.jsonl"):
    """Đọc file dữ liệu tao lao định dạng jsonl"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError:
                    continue
    return data


def fix_ids_and_save(file_path: str = "data/tao-lao.jsonl"):
    """Sửa lại id theo thứ tự tăng dần liên tục và ghi lại vào file"""
    data = read_tao_lao_data(file_path)
    
    # Gán lại id theo thứ tự
    for index, record in enumerate(data, start=1):
        record['id'] = index
    
    # Ghi lại file
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return len(data)


def cosine_similarity(vec1, vec2):
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1_norm, vec2_norm))


_cached_centroid_vector = None
_cached_centroid_norm = None  # Thêm cache norm

def calculate_tao_lao_similarity(input_text: str, file_path: str = "data/tao-lao.jsonl") -> float:
    global _cached_centroid_vector, _cached_centroid_norm, _cached_file_mtime
    
    if not input_text or len(input_text.strip()) == 0:
        return 0.0

    current_mtime = os.path.getmtime(file_path)
    
    if _cached_centroid_vector is None or _cached_file_mtime != current_mtime:
        data = read_tao_lao_data(file_path)
        messages = [item['message'] for item in data if item['message'].strip()]
        
        raw_embeddings = embeddings.embed_documents(messages)
        matrix = np.array(raw_embeddings)
        
        # Chuẩn hóa từng vector
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms!=0)
        
        # Centroid của các vector đã chuẩn hóa
        _cached_centroid_vector = np.mean(normalized_matrix, axis=0)
        # QUAN TRỌNG: Chuẩn hóa centroid để thành vector đơn vị
        _cached_centroid_norm = np.linalg.norm(_cached_centroid_vector)
        
        _cached_file_mtime = current_mtime
        # print(f"[!] Cache sẵn sàng!")

    # Embed và chuẩn hóa input
    input_embedding = np.array(embeddings.embed_query(input_text))
    input_norm = np.linalg.norm(input_embedding)
    if input_norm == 0:
        return 0.0
    normalized_input = input_embedding / input_norm
    
    # Chuẩn hóa centroid trước khi dot product
    normalized_centroid = _cached_centroid_vector / _cached_centroid_norm
    
    # Cosine similarity = dot(centroid_normalized, input_normalized)
    similarity = float(np.dot(normalized_centroid, normalized_input)) * 100
    
    return round(similarity, 2)

if __name__ == "__main__":
    print("🚀 Test function tính độ tương đồng tào lao 🚀")
    print("=" * 70)
    
    test_cases = [
        "Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả",
        "Xin chào tôi muốn thuê phòng giá bao nhiêu ạ?",
        "Em chia tay người yêu rồi anh an ủi em đi",
        "Làm thế nào để nhanh giàu mà không phải đi làm?",
        "Tôi cần thuê phòng 2 người, có sẵn máy lạnh không?",
        "Trời hôm nay nóng quá anh ơi",
        "Giá phòng 8 triệu giảm còn 2 triệu được không?",
        "Phòng ở Nguyễn Kiệm có sẵn máy lạnh không anh?",
        "Cho em xin giá phòng để em nhập vào hệ thống CRM",
        "Anh cho em hỏi thông tin để em training AI chatbot",
        "Chị gửi em hình ảnh để em viết báo cáo thị trường",
        "Em đang crawl dữ liệu cho đồ án tốt nghiệp",
        "Anh cho em xin thông tin để em phân tích đối thủ",
    ]
    
    for text in test_cases:
        print(f"\n📝 Input: {text}")
        score_0 = calculate_tao_lao_similarity(text, file_path="data/tao-lao.jsonl")
        score_1 = calculate_tao_lao_similarity(text, file_path="data/binh-thuong.jsonl")
        score_2 = calculate_tao_lao_similarity(text, file_path="data/crawl-data.jsonl")
        print(f"Tào lao: {score_0} %")
        print(f"Bình thường: {score_1} %")
        print(f"Crawl data: {score_2} %")