import os
import json
import numpy as np
import requests

from langchain_openai import OpenAIEmbeddings
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

embeddings = OpenAIEmbeddings(
    model=model_id,
    base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    check_embedding_ctx_length=False,
)

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
    messages = [item['message'] for item in data if item.get('message', '').strip()]
    
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

def predict_3_class(input_text: str):
    """Dự đoán lớp có similarity cao nhất"""
    classes = {
        "TAO_LAO": "data/tao-lao.jsonl",
        "BINH_THUONG": "data/binh-thuong.jsonl",
        "CRAWL_DATA": "data/crawl-data.jsonl"
    }
    
    scores = {}
    for label, file_path in classes.items():
        centroid, _ = get_centroid(file_path)
        scores[label] = cosine_similarity_with_centroid(input_text, centroid)
    
    # Argmax - chọn lớp có điểm cao nhất
    predicted_label = max(scores, key=scores.get)
    
    return predicted_label, scores

if __name__ == "__main__":
    print("🚀 Đánh giá Accuracy 3 lớp: TAO_LAO | BINH_THUONG | CRAWL_DATA")
    print("=" * 70)
    
    # Test cases: (text, true_label)
    test_cases = [
        # === TÀO LAO (15 samples) ===
        ("Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả", "TAO_LAO"),
        ("Em sale dễ thương quá, tối nay có rảnh đi cà phê với anh không?", "TAO_LAO"),
        ("Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ vài tháng nhé?", "TAO_LAO"),
        ("Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền anh đi!", "TAO_LAO"),
        ("Bên em có bán kem trộn không, dạo này da anh đen quá.", "TAO_LAO"),
        ("Nay xổ số miền Bắc đánh con gì dễ trúng hả em?", "TAO_LAO"),
        ("Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt cọc luôn.", "TAO_LAO"),
        ("Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn nhé?", "TAO_LAO"),
        ("Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng được không?", "TAO_LAO"),
        ("Em ăn cơm chưa?", "TAO_LAO"),
        ("Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi em.", "TAO_LAO"),
        ("Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh ngay!", "TAO_LAO"),
        ("Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.", "TAO_LAO"),
        ("Phòng này bao ăn bao ở bao luôn cả người yêu không em?", "TAO_LAO"),
        
        # === BÌNH THƯỜNG (15 samples) ===
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
        
        # === CRAWL DATA (15 samples) ===
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
        ("Cho em xem ảnh thực tế để em check chất lượng", "CRAWL_DATA"),
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
    
    print(f"\n🔍 Bắt đầu test {total} samples...\n")
    
    for i, (text, true_label) in enumerate(test_cases, 1):
        pred_label, scores = predict_3_class(text)
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        confusion[true_label][pred_label] += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} #{i:02d} [{pred_label:12s}] Tào:{scores['TAO_LAO']:5.1f}% "
              f"Bình:{scores['BINH_THUONG']:5.1f}% Crawl:{scores['CRAWL_DATA']:5.1f}%")
        print(f"      Text: {text[:55]}{'...' if len(text) > 55 else ''}")
    
    # Kết quả
    accuracy = correct / total * 100
    
    print(f"\n{'='*70}")
    print(f"📊 KẾT QUẢ TỔNG HỢP")
    print(f"{'='*70}")
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"\n📋 Confusion Matrix:")
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
        
        
# Auto-detected model: Qwen/Qwen3-Embedding-0.6B
# 🚀 Đánh giá Accuracy 3 lớp: TAO_LAO | BINH_THUONG | CRAWL_DATA
# ======================================================================

# 🔍 Bắt đầu test 45 samples...

# 📊 Computing centroid for data/tao-lao.jsonl (590 samples)...
# 📊 Computing centroid for data/binh-thuong.jsonl (523 samples)...
# 📊 Computing centroid for data/crawl-data.jsonl (300 samples)...
# ✅ #01 [TAO_LAO     ] Tào: 57.2% Bình: 46.3% Crawl: 45.9%
#       Text: Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả
# ✅ #02 [TAO_LAO     ] Tào: 56.3% Bình: 48.7% Crawl: 39.1%
#       Text: Em sale dễ thương quá, tối nay có rảnh đi cà phê với an...
# ✅ #03 [TAO_LAO     ] Tào: 66.4% Bình: 63.4% Crawl: 49.1%
#       Text: Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ ...
# ✅ #04 [TAO_LAO     ] Tào: 46.4% Bình: 37.3% Crawl: 29.1%
#       Text: Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền a...
# ✅ #05 [TAO_LAO     ] Tào: 51.6% Bình: 42.2% Crawl: 34.8%
#       Text: Bên em có bán kem trộn không, dạo này da anh đen quá.
# ✅ #06 [TAO_LAO     ] Tào: 38.6% Bình: 35.1% Crawl: 28.0%
#       Text: Nay xổ số miền Bắc đánh con gì dễ trúng hả em?
# ✅ #07 [TAO_LAO     ] Tào: 34.6% Bình: 24.6% Crawl: 23.8%
#       Text: Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt ...
# ✅ #08 [TAO_LAO     ] Tào: 52.8% Bình: 48.1% Crawl: 36.6%
#       Text: Căn này nhìn phong thủy u ám quá, có ma không em?
# ❌ #09 [BINH_THUONG ] Tào: 55.3% Bình: 56.9% Crawl: 45.2%
#       Text: Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn...
# ✅ #10 [TAO_LAO     ] Tào: 60.8% Bình: 48.6% Crawl: 42.5%
#       Text: Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng...
# ✅ #11 [TAO_LAO     ] Tào: 49.8% Bình: 49.8% Crawl: 40.7%
#       Text: Em ăn cơm chưa?
# ✅ #12 [TAO_LAO     ] Tào: 54.6% Bình: 45.0% Crawl: 41.2%
#       Text: Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi e...
# ✅ #13 [TAO_LAO     ] Tào: 51.2% Bình: 44.0% Crawl: 36.0%
#       Text: Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh n...
# ✅ #14 [TAO_LAO     ] Tào: 48.8% Bình: 48.1% Crawl: 37.2%
#       Text: Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.
# ❌ #15 [BINH_THUONG ] Tào: 56.6% Bình: 57.0% Crawl: 36.2%
#       Text: Phòng này bao ăn bao ở bao luôn cả người yêu không em?
# ✅ #16 [BINH_THUONG ] Tào: 49.9% Bình: 58.2% Crawl: 52.0%
#       Text: Căn hộ này giá thuê một tháng là bao nhiêu vậy em?
# ✅ #17 [BINH_THUONG ] Tào: 56.3% Bình: 58.8% Crawl: 44.3%
#       Text: Em cho anh xin thêm hình ảnh thật của căn 1 phòng ngủ n...
# ✅ #18 [BINH_THUONG ] Tào: 39.3% Bình: 47.2% Crawl: 36.6%
#       Text: Địa chỉ chính xác của tòa nhà này ở đâu em?
# ✅ #19 [BINH_THUONG ] Tào: 54.0% Bình: 64.4% Crawl: 53.4%
#       Text: Giá thuê này đã bao gồm phí quản lý và dọn phòng chưa?
# ✅ #20 [BINH_THUONG ] Tào: 37.9% Bình: 42.0% Crawl: 40.3%
#       Text: Tiền điện nước ở đây tính theo giá nhà nước hay giá dịc...
# ✅ #21 [BINH_THUONG ] Tào: 57.7% Bình: 64.0% Crawl: 48.2%
#       Text: Bên mình có căn nào full nội thất xách vali vào ở luôn ...
# ✅ #22 [BINH_THUONG ] Tào: 44.8% Bình: 47.9% Crawl: 42.3%
#       Text: Có chỗ đậu xe ô tô không em, phí gửi xe tháng bao nhiêu...
# ✅ #23 [BINH_THUONG ] Tào: 42.4% Bình: 48.4% Crawl: 31.7%
#       Text: Khu vực này có hay bị ngập nước vào mùa mưa không em?
# ✅ #24 [BINH_THUONG ] Tào: 48.2% Bình: 57.5% Crawl: 44.5%
#       Text: Tòa nhà mình có thang máy và bảo vệ 24/7 không?
# ✅ #25 [BINH_THUONG ] Tào: 58.4% Bình: 58.9% Crawl: 39.8%
#       Text: Anh muốn đi xem phòng thực tế vào chiều nay có được khô...
# ✅ #26 [BINH_THUONG ] Tào: 47.4% Bình: 52.7% Crawl: 47.1%
#       Text: Căn studio diện tích bao nhiêu mét vuông vậy em?
# ✅ #27 [BINH_THUONG ] Tào: 50.7% Bình: 58.0% Crawl: 37.6%
#       Text: Có được nấu ăn trong phòng không em? Bếp điện hay gas?
# ✅ #28 [BINH_THUONG ] Tào: 46.6% Bình: 51.0% Crawl: 36.7%
#       Text: Xung quanh đây có siêu thị hay cửa hàng tiện lợi nào gầ...
# ✅ #29 [BINH_THUONG ] Tào: 49.6% Bình: 58.4% Crawl: 38.2%
#       Text: Căn 2 phòng ngủ có 2 nhà vệ sinh riêng biệt không em?
# ❌ #30 [TAO_LAO     ] Tào: 44.3% Bình: 40.4% Crawl: 39.7%
#       Text: Nếu anh ký hợp đồng dài hạn 1 năm thì có được giảm giá ...
# ❌ #31 [BINH_THUONG ] Tào: 64.4% Bình: 66.6% Crawl: 63.2%
#       Text: Cho em xin giá phòng để em nhập vào hệ thống CRM
# ✅ #32 [CRAWL_DATA  ] Tào: 52.2% Bình: 42.8% Crawl: 53.3%
#       Text: Anh cho em hỏi thông tin để em training AI chatbot
# ✅ #33 [CRAWL_DATA  ] Tào: 44.7% Bình: 37.2% Crawl: 49.5%
#       Text: Chị gửi em hình ảnh để em viết báo cáo thị trường
# ✅ #34 [CRAWL_DATA  ] Tào: 50.6% Bình: 44.8% Crawl: 52.6%
#       Text: Em đang crawl dữ liệu cho đồ án tốt nghiệp
# ✅ #35 [CRAWL_DATA  ] Tào: 58.7% Bình: 50.5% Crawl: 67.3%
#       Text: Anh cho em xin thông tin để em phân tích đối thủ
# ❌ #36 [TAO_LAO     ] Tào: 53.2% Bình: 46.7% Crawl: 52.6%
#       Text: Em cần data để train model machine learning
# ✅ #37 [CRAWL_DATA  ] Tào: 55.1% Bình: 51.0% Crawl: 61.5%
#       Text: Cho em xem giá tham khảo để em viết bài review
# ❌ #38 [BINH_THUONG ] Tào: 50.1% Bình: 55.6% Crawl: 44.8%
#       Text: Em đang làm research về thị trường nhà trọ quận này
# ✅ #39 [CRAWL_DATA  ] Tào: 57.5% Bình: 48.9% Crawl: 62.3%
#       Text: Anh cho em xin thông tin để em nhập database
# ✅ #40 [CRAWL_DATA  ] Tào: 55.1% Bình: 48.3% Crawl: 55.9%
#       Text: Em cần mẫu tin nhắn để xây dựng chatbot tự động
# ❌ #41 [TAO_LAO     ] Tào: 53.1% Bình: 46.4% Crawl: 44.0%
#       Text: Cho em xem ảnh thực tế để em check chất lượng
# ✅ #42 [CRAWL_DATA  ] Tào: 45.5% Bình: 44.0% Crawl: 48.5%
#       Text: Em đang thu thập dữ liệu cho luận văn master
# ✅ #43 [CRAWL_DATA  ] Tào: 60.8% Bình: 52.4% Crawl: 66.2%
#       Text: Anh gửi em bảng giá để em so sánh với đối thủ
# ✅ #44 [CRAWL_DATA  ] Tào: 58.8% Bình: 52.6% Crawl: 62.9%
#       Text: Em cần thông tin để populate vào app của em
# ✅ #45 [CRAWL_DATA  ] Tào: 53.5% Bình: 44.4% Crawl: 57.1%
#       Text: Cho em hỏi giá để em scraping data xây dựng API

# ======================================================================
# 📊 KẾT QUẢ TỔNG HỢP
# ======================================================================
# ✅ Accuracy: 84.44% (38/45)

# 📋 Confusion Matrix:
#                 |   Pred Tào |  Pred Bình | Pred Crawl
# ------------------------------------------------------------
# TAO_LAO         |         13 |          2 |          0
# BINH_THUONG     |          1 |         14 |          0
# CRAWL_DATA      |          2 |          2 |         11

# 📈 Per-class Accuracy:
#   TAO_LAO     : 86.67% (13/15)
#   BINH_THUONG : 93.33% (14/15)
#   CRAWL_DATA  : 73.33% (11/15)