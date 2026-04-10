import os
import json
import numpy as np
import requests

from langchain_openai import OpenAIEmbeddings

class FPTEmbeddings:
    def __init__(self, api_key, model="multilingual-e5-large"):
        self.url = "https://mkp-api.fptcloud.com/v1/embeddings"
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:  # ⚠️ gửi từng câu để tránh 422
            res = requests.post(
                self.url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": text
                }
            )

            if res.status_code != 200:
                raise Exception(f"API Error: {res.status_code} - {res.text}")

            data = res.json()

            try:
                embeddings.append(data["data"][0]["embedding"])
            except Exception:
                raise Exception(f"Bad response format: {data}")

        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()

# Config
base_url = os.getenv("OPENAI_BASE_URL_EMBED")
try:
    resp = requests.get(f"{base_url}/models", timeout=5)
    model_id = resp.json()["data"][0]["id"]
    print(f"🔍 Auto-detected model: {model_id}")
except Exception:
    model_id = os.getenv("OPENAI_API_MODEL_NAME_EMBED")
    print(f"⚠️ Using fallback model: {model_id}")

# embeddings = FPTEmbeddings(
#     api_key="",
#     model=model_id
# )

# 🔥 FIX 2: Bỏ chữ ".." đi, vì file model đang nằm CÙNG MỘT CHỖ với file router.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "router_model.pkl")

embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    # base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    # api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    base_url="https://api.novita.ai/openai",
    api_key=SecretStr(os.getenv("NOVITA_API_KEY", "text")),
    check_embedding_ctx_length=False,
    chunk_size=32
    
)

# print(len(embeddings.embed_query("Test embedding")))  # Test kết nối

# Cache cho 3 lớp: {file_path: (centroid_vector, norm, mtime)}
_centroid_cache = {}

def read_jsonl(file_path: str):
    """Đọc file jsonl"""
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

# def get_centroid(label, file_path, outlier_percentile=10):
#     """
#     Tính vector đại diện tối ưu bằng cách loại bỏ nhiễu và outliers.
#     outlier_percentile: % những câu 'lạc quẻ' nhất sẽ bị loại bỏ (mặc định 10%).
#     """
#     global _centroid_cache
    
#     current_mtime = os.path.getmtime(file_path)
#     if label in _centroid_cache:
#         cached_vec, cached_mtime = _centroid_cache[label]
#         if cached_mtime == current_mtime:
#             return cached_vec

#     print(f"🚀 Đang tinh chỉnh Vector đại diện cho lớp: {label}...")
    
#     texts = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip(): continue
#             try:
#                 record = json.loads(line)
#                 text = record.get("conversation", "").lower().strip()
                
#                 # CẢI TIẾN 1: Lọc dữ liệu thô
#                 # Loại bỏ câu quá ngắn (nhiễu) vì chúng không đủ thông tin ngữ nghĩa
#                 if len(text.split()) > 3: 
#                     texts.append(text)
#             except: continue

#     if not texts: 
#         return np.zeros(1024)

#     # Nhúng văn bản
#     raw_vecs = np.array(embeddings.embed_documents(texts))
    
#     # CẢI TIẾN 2: Chuẩn hóa Unit Vector trước khi xử lý
#     norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
#     matrix = raw_vecs / np.where(norms == 0, 1, norms)

#     # CẢI TIẾN 3: Loại bỏ Outliers (Những câu nằm quá xa tâm điểm)
#     if len(matrix) > 5:  # Chỉ lọc nếu có đủ dữ liệu
#         # Tính tâm tạm thời
#         initial_centroid = np.mean(matrix, axis=0)
#         # Tính khoảng cách Cosine của từng câu so với tâm này
#         # (Vì đã chuẩn hóa nên khoảng cách tỉ lệ thuận với hiệu vector)
#         distances = np.linalg.norm(matrix - initial_centroid, axis=1)
        
#         # Xác định ngưỡng (ví dụ: loại bỏ 10% xa nhất)
#         threshold = np.percentile(distances, 100 - outlier_percentile)
#         filtered_matrix = matrix[distances <= threshold]
        
#         print(f"   ↳ Đã loại bỏ {len(matrix) - len(filtered_matrix)} câu nhiễu.")
#     else:
#         filtered_matrix = matrix

#     # CẢI TIẾN 4: Tính Centroid cuối cùng
#     centroid = np.mean(filtered_matrix, axis=0)
#     # Chuẩn hóa lại lần cuối để đảm bảo độ dài bằng 1
#     centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    
#     _centroid_cache[label] = (centroid, current_mtime)
#     return centroid

def get_centroid(file_path: str):
    """Lấy centroid cho một file, có cache"""
    global _centroid_cache
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy {file_path}")
    
    current_mtime = os.path.getmtime(file_path)
    
    if file_path in _centroid_cache:
        vec, norm, mtime = _centroid_cache[file_path]
        if mtime == current_mtime:
            return vec, norm
    
    # Tính centroid mới
    data = read_jsonl(file_path)
    messages = [item['conversation'] for item in data if item.get('conversation', '').strip()]
    if not messages:
        raise ValueError(f"{file_path} không có dữ liệu")
    
    print(f"📊 Computing centroid for {file_path} ({len(messages)} samples)...")
    
    # Batch embedding (TEI giới hạn 32)
    BATCH_SIZE = 32
    raw_embeddings = []
    
    for i in range(0, len(messages), BATCH_SIZE):
        batch = messages[i:i+BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch)
        raw_embeddings.extend(batch_embeddings)
    
    matrix = np.array(raw_embeddings)
    
    # Chuẩn hóa từng vector thành unit vector
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized_matrix = matrix / norms
    
    # Centroid = trung bình các unit vectors
    centroid = np.mean(normalized_matrix, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    
    _centroid_cache[file_path] = (centroid, centroid_norm, current_mtime)
    return centroid, centroid_norm


def cosine_similarity_with_centroid(input_text: str, centroid: np.ndarray):
    """Tính cosine similarity giữa input và centroid"""
    if not input_text or not input_text.strip():
        return 0.0
    
    input_embedding = np.array(embeddings.embed_query(input_text))
    input_norm = np.linalg.norm(input_embedding)
    
    if input_norm == 0:
        return 0.0
    
    normalized_input = input_embedding / input_norm
    similarity = float(np.dot(centroid, normalized_input))
    
    return similarity * 100  # Trả về %

classes = {
    "TAO_LAO": "data/tao-lao.jsonl",
    "BINH_THUONG": "data/binh-thuong.jsonl",
    "CRAWL_DATA": "data/crawl-data.jsonl"
}

def predict_3_class(input_text: str):
    """
    So sánh thuần túy giữa vector input và 3 vector đại diện lớp.
    """
    if not input_text or not input_text.strip():
        return "BINH_THUONG", {"BINH_THUONG": 100.0}

    # 1. Nhúng input và chuẩn hóa
    input_vec = np.array(embeddings.embed_query(input_text.lower().strip()))
    # print(len(embeddings.embed_query(input_text.lower().strip())))
    input_vec = input_vec / np.linalg.norm(input_vec) # Đảm bảo tổng bình p luôn bằng 1 để tính cosine similarity chính xác

    scores = {}
    for label, path in classes.items():
        # 2. Lấy centroid của lớp
        centroid_vec = get_centroid(label, path)
        
        # 3. Tính Cosine Similarity (Tích vô hướng của 2 unit vectors)
        # Giá trị từ -1 đến 1 -> Quy đổi ra % (0-100)
        similarity = np.dot(input_vec, centroid_vec)
        scores[label] = round(max(0, similarity) * 100, 2)

    # 4. Lấy Top-1 cao nhất
    predicted_label = max(scores, key=scores.get)
    
    return predicted_label, scores

if __name__ == "__main__":
    print("🚀 Đánh giá Accuracy 3 lớp: TAO_LAO | BINH_THUONG | CRAWL_DATA")
    print("=" * 70)
    
    # Test cases: (text, true_label)
    test_cases = [
        # === TÀO LAO (12 samples) ===
        ("Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả", "TAO_LAO"),
        ("Em sale dễ thương quá, tối nay có rảnh đi cà phê với anh không?", "TAO_LAO"),
        ("Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ vài tháng nhé?", "TAO_LAO"),
        ("Bên em có bán kem trộn không, dạo này da anh đen quá.", "TAO_LAO"),
        ("Nay xổ số miền Bắc đánh con gì dễ trúng hả em?", "TAO_LAO"),
        ("Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt cọc luôn.", "TAO_LAO"),
        ("Căn này nhìn phong thủy u ám quá, có ma không em?", "TAO_LAO"),
        ("Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn nhé?", "TAO_LAO"),
        ("Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng được không?", "TAO_LAO"),
        ("Em ăn cơm chưa?", "TAO_LAO"),
        ("Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi em.", "TAO_LAO"),
        ("Phòng này bao ăn bao ở bao luôn cả người yêu không em?", "TAO_LAO"),
        ("Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.", "TAO_LAO"),
        
        # === BÌNH THƯỜNG (18 samples) ===
        ("Căn hộ này giá thuê một tháng là bao nhiêu vậy em?", "BINH_THUONG"),
        ("Em cho anh xin thêm hình ảnh thật của căn 1 phòng ngủ nhé.", "BINH_THUONG"),
        ("Địa chỉ chính xác của tòa nhà này ở đâu em?", "BINH_THUONG"),
        ("Giá thuê này đã bao gồm phí quản lý và dọn phòng chưa?", "BINH_THUONG"),
        ("Tiền điện nước ở đây tính theo giá nhà nước hay giá dịch vụ?", "BINH_THUONG"),
        ("Bên mình có căn nào full nội thất xách vali vào ở luôn không?", "BINH_THUONG"),
        ("Có chỗ đậu xe ô tô không em, phí gửi xe tháng bao nhiêu?", "BINH_THUONG"),
        ("Khu vực này có hay bị ngập nước vào mùa mưa không em?", "BINH_THUONG"),
        ("Tòa nhà mình có thang máy và bảo vệ 24/7 không?", "BINH_THUONG"),
        ("Anh muốn đi xem phòng thực tế vào chiều nay có được không?", "BINH_THUONG"),
        ("Căn studio diện tích bao nhiêu mét vuông vậy em?", "BINH_THUONG"),
        ("Có được nấu ăn trong phòng không em? Bếp điện hay gas?", "BINH_THUONG"),
        ("Xung quanh đây có siêu thị hay cửa hàng tiện lợi nào gần không?", "BINH_THUONG"),
        ("Căn 2 phòng ngủ có 2 nhà vệ sinh riêng biệt không em?", "BINH_THUONG"),
        ("Nếu anh ký hợp đồng dài hạn 1 năm thì có được giảm giá không?", "BINH_THUONG"),
        ("Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền anh đi!", "BINH_THUONG"),
        ("Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh ngay!", "BINH_THUONG"),
        
        ("Cho em xem ảnh thực tế để em check chất lượng", "TAO_LAO"),
        # === CRAWL DATA (14 samples) ===
        ("Cho em xin giá phòng để em nhập vào hệ thống CRM", "CRAWL_DATA"),
        ("Anh cho em hỏi thông tin để em training AI chatbot", "CRAWL_DATA"),
        ("Chị gửi em hình ảnh để em viết báo cáo thị trường", "CRAWL_DATA"),
        ("Em đang crawl dữ liệu cho đồ án tốt nghiệp", "CRAWL_DATA"),
        ("Anh cho em xin thông tin để em phân tích đối thủ", "CRAWL_DATA"),
        ("Em cần data để train model machine learning", "CRAWL_DATA"),
        ("Cho em xem giá tham khảo để em viết bài review", "CRAWL_DATA"),
        ("Em đang làm research về thị trường nhà trọ quận này", "CRAWL_DATA"),
        ("Anh cho em xin thông tin để em nhập database", "CRAWL_DATA"),
        ("Em cần mẫu tin nhắn để xây dựng chatbot tự động", "CRAWL_DATA"),
        ("Em đang thu thập dữ liệu cho luận văn master", "CRAWL_DATA"),
        ("Anh gửi em bảng giá để em so sánh với đối thủ", "CRAWL_DATA"),
        ("Em cần thông tin để populate vào app của em", "CRAWL_DATA"),
        ("Cho em hỏi giá để em scraping data xây dựng API", "CRAWL_DATA"),
    ]
    
    # Đánh giá
    correct = 0
    total = len(test_cases)
    
    # Confusion matrix: {true: {pred: count}}
    labels = ["TAO_LAO", "BINH_THUONG", "CRAWL_DATA"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    
    print(f"\n🔍 Bắt đầu test {total} samples bằng thuật toán Centroid...\n")
    
    for i, (text, true_label) in enumerate(test_cases, 1):
        # 🔥 FIX 1: Dùng text thuần túy
        pred_label, scores = predict_3_class(text)
        
        if pred_label not in labels:
            pred_label = "BINH_THUONG"
            
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
            
        confusion[true_label][pred_label] += 1
        
        # 🔥 FIX 2: Lấy điểm an toàn
        score_tao_lao = scores.get('TAO_LAO', 0.0)
        score_binh_thuong = scores.get('BINH_THUONG', 0.0)
        score_crawl = scores.get('CRAWL_DATA', 0.0)
        
        status = "✅" if is_correct else "❌"
        print(f"  Text: {text}")
        print(f"{status} #{i:02d} [{pred_label:12s}] Tào:{score_tao_lao:5.1f}% "
              f"Bình:{score_binh_thuong:5.1f}% Crawl:{score_crawl:5.1f}%\n")
    
    # Kết quả
    accuracy = correct / total * 100
    
    print(f"{'='*70}")
    print(f"📊 KẾT QUẢ TỔNG HỢP (CENTROID METHOD)")
    print(f"{'='*70}")
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"\n📋 Confusion Matrix (Hàng: Thực tế | Cột: AI Dự đoán):")
    print(f"{'':15} | {'Pred Tào':>10} | {'Pred Bình':>10} | {'Pred Crawl':>10}")
    print("-" * 60)
    for true in labels:
        row = confusion[true]
        print(f"{true:15} | {row['TAO_LAO']:10} | {row['BINH_THUONG']:10} | {row['CRAWL_DATA']:10}")
    
    # Per-class accuracy
    print(f"\n📈 Per-class Accuracy:")
    for label in labels:
        tp = confusion[label][label]
        total_class = sum(confusion[label].values())
        acc = tp / total_class * 100 if total_class > 0 else 0
        print(f"  {label:12}: {acc:5.2f}% ({tp}/{total_class})")