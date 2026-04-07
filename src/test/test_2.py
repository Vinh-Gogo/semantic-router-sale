import os
import json
import requests
import numpy as np

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query theo chuẩn Qwen3 embedding"""
    return f'Instruct: {task_description}\nQuery: {query}'


# Task instruction cho Qwen3-Embedding
TASK_INSTRUCTION = 'You are a Sales Assistant. retrieve any messages related to the customer\'s intention to rent a serviced apartment.'

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
    """Đọc file jsonl trả về list messages"""
    messages = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        msg = record.get("message", "").strip()
                        if msg:
                            messages.append(msg)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return messages


def get_centroid(file_path: str):
    """Lấy centroid cho một lớp, có cache"""
    global _centroid_cache
    
    current_mtime = os.path.getmtime(file_path)
    
    # Kiểm tra cache
    if file_path in _centroid_cache:
        vec, norm, mtime = _centroid_cache[file_path]
        if mtime == current_mtime:
            return vec, norm
    
    # Tính centroid mới
    messages = read_jsonl(file_path)
    if not messages:
        raise ValueError(f"{file_path} không có dữ liệu")
    
    print(f"📊 Computing centroid for {file_path} ({len(messages)} samples)...")
    
    # ❗ QUAN TRỌNG: Documents KHÔNG thêm instruction (theo chuẩn Qwen3)
    raw_embeddings = embeddings.embed_documents(messages)
    matrix = np.array(raw_embeddings)
    
    # Chuẩn hóa từng vector thành unit vector
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Tránh chia cho 0
    unit_vectors = matrix / norms
    
    # Centroid = trung bình các unit vectors
    centroid = np.mean(unit_vectors, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    
    if centroid_norm > 0:
        centroid = centroid / centroid_norm  # Chuẩn hóa centroid
    
    # Lưu cache
    _centroid_cache[file_path] = (centroid, centroid_norm, current_mtime)
    return centroid, centroid_norm


def predict_3_class(input_text: str):
    """Dự đoán lớp có similarity cao nhất"""
    classes = {
        "TAO_LAO": "data/tao-lao.jsonl",
        "BINH_THUONG": "data/binh-thuong.jsonl",
        "CRAWL_DATA": "data/crawl-data.jsonl"
    }
    
    # ❗ QUAN TRỌNG: Query PHẢI thêm instruction (theo chuẩn Qwen3)
    wrapped_input = get_detailed_instruct(TASK_INSTRUCTION, input_text)
    input_embedding = np.array(embeddings.embed_query(wrapped_input))
    input_norm = np.linalg.norm(input_embedding)
    
    if input_norm == 0:
        return "UNKNOWN", {label: 0.0 for label in classes}
    
    normalized_input = input_embedding / input_norm
    
    scores = {}
    for label, file_path in classes.items():
        centroid, _ = get_centroid(file_path)
        similarity = float(np.dot(centroid, normalized_input))
        scores[label] = similarity * 100  # Convert to %
    
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
        ("Căn này nhìn phong thủy u ám quá, có ma không em?", "TAO_LAO"),
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


# 🔍 Auto-detected model: Qwen/Qwen3-Embedding-0.6B
# 🚀 Đánh giá Accuracy 3 lớp: TAO_LAO | BINH_THUONG | CRAWL_DATA
# ======================================================================

# 🔍 Bắt đầu test 45 samples...

# 📊 Computing centroid for data/tao-lao.jsonl (590 samples)...
# 📊 Computing centroid for data/binh-thuong.jsonl (523 samples)...
# 📊 Computing centroid for data/crawl-data.jsonl (300 samples)...
# ✅ #01 [TAO_LAO     ] Tào: 59.8% Bình: 55.6% Crawl: 53.0%
#       Text: Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả
# ✅ #02 [TAO_LAO     ] Tào: 56.9% Bình: 56.7% Crawl: 49.2%
#       Text: Em sale dễ thương quá, tối nay có rảnh đi cà phê với an...
# ✅ #03 [TAO_LAO     ] Tào: 60.1% Bình: 60.1% Crawl: 53.5%
#       Text: Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ ...
# ❌ #04 [BINH_THUONG ] Tào: 57.9% Bình: 59.5% Crawl: 48.0%
#       Text: Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền a...
# ✅ #05 [TAO_LAO     ] Tào: 51.7% Bình: 45.7% Crawl: 42.1%
#       Text: Bên em có bán kem trộn không, dạo này da anh đen quá.
# ✅ #06 [TAO_LAO     ] Tào: 48.5% Bình: 46.0% Crawl: 41.5%
#       Text: Nay xổ số miền Bắc đánh con gì dễ trúng hả em?
# ✅ #07 [TAO_LAO     ] Tào: 56.0% Bình: 54.4% Crawl: 48.7%
#       Text: Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt ...
# ❌ #08 [BINH_THUONG ] Tào: 56.5% Bình: 57.3% Crawl: 47.0%
#       Text: Căn này nhìn phong thủy u ám quá, có ma không em?
# ❌ #09 [BINH_THUONG ] Tào: 53.0% Bình: 55.5% Crawl: 46.3%
#       Text: Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn...
# ✅ #10 [TAO_LAO     ] Tào: 56.9% Bình: 50.8% Crawl: 45.1%
#       Text: Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng...
# ❌ #11 [BINH_THUONG ] Tào: 58.9% Bình: 61.9% Crawl: 54.1%
#       Text: Em ăn cơm chưa?
# ✅ #12 [TAO_LAO     ] Tào: 57.5% Bình: 55.0% Crawl: 50.0%
#       Text: Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi e...
# ❌ #13 [BINH_THUONG ] Tào: 52.9% Bình: 54.1% Crawl: 48.0%
#       Text: Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh n...
# ❌ #14 [BINH_THUONG ] Tào: 54.2% Bình: 55.7% Crawl: 43.8%
#       Text: Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.
# ❌ #15 [BINH_THUONG ] Tào: 57.6% Bình: 61.0% Crawl: 48.0%
#       Text: Phòng này bao ăn bao ở bao luôn cả người yêu không em?
# ✅ #16 [BINH_THUONG ] Tào: 51.8% Bình: 58.8% Crawl: 55.8%
#       Text: Căn hộ này giá thuê một tháng là bao nhiêu vậy em?
# ✅ #17 [BINH_THUONG ] Tào: 56.3% Bình: 60.4% Crawl: 50.0%
#       Text: Em cho anh xin thêm hình ảnh thật của căn 1 phòng ngủ n...
# ✅ #18 [BINH_THUONG ] Tào: 44.2% Bình: 52.2% Crawl: 43.5%
#       Text: Địa chỉ chính xác của tòa nhà này ở đâu em?
# ✅ #19 [BINH_THUONG ] Tào: 58.4% Bình: 66.7% Crawl: 59.3%
#       Text: Giá thuê này đã bao gồm phí quản lý và dọn phòng chưa?
# ✅ #20 [BINH_THUONG ] Tào: 45.6% Bình: 52.3% Crawl: 50.9%
#       Text: Tiền điện nước ở đây tính theo giá nhà nước hay giá dịc...
# ✅ #21 [BINH_THUONG ] Tào: 57.6% Bình: 62.4% Crawl: 52.4%
#       Text: Bên mình có căn nào full nội thất xách vali vào ở luôn ...
# ✅ #22 [BINH_THUONG ] Tào: 49.9% Bình: 54.5% Crawl: 49.4%
#       Text: Có chỗ đậu xe ô tô không em, phí gửi xe tháng bao nhiêu...
# ✅ #23 [BINH_THUONG ] Tào: 52.8% Bình: 60.4% Crawl: 46.0%
#       Text: Khu vực này có hay bị ngập nước vào mùa mưa không em?
# ✅ #24 [BINH_THUONG ] Tào: 52.1% Bình: 61.7% Crawl: 49.4%
#       Text: Tòa nhà mình có thang máy và bảo vệ 24/7 không?
# ✅ #25 [BINH_THUONG ] Tào: 58.3% Bình: 62.5% Crawl: 47.8%
#       Text: Anh muốn đi xem phòng thực tế vào chiều nay có được khô...
# ✅ #26 [BINH_THUONG ] Tào: 50.7% Bình: 56.5% Crawl: 49.5%
#       Text: Căn studio diện tích bao nhiêu mét vuông vậy em?
# ✅ #27 [BINH_THUONG ] Tào: 53.6% Bình: 62.2% Crawl: 46.8%
#       Text: Có được nấu ăn trong phòng không em? Bếp điện hay gas?
# ✅ #28 [BINH_THUONG ] Tào: 49.1% Bình: 53.9% Crawl: 41.5%
#       Text: Xung quanh đây có siêu thị hay cửa hàng tiện lợi nào gầ...
# ✅ #29 [BINH_THUONG ] Tào: 53.8% Bình: 63.0% Crawl: 50.1%
#       Text: Căn 2 phòng ngủ có 2 nhà vệ sinh riêng biệt không em?
# ✅ #30 [BINH_THUONG ] Tào: 46.9% Bình: 48.3% Crawl: 46.8%
#       Text: Nếu anh ký hợp đồng dài hạn 1 năm thì có được giảm giá ...
# ✅ #31 [CRAWL_DATA  ] Tào: 55.4% Bình: 60.5% Crawl: 64.8%
#       Text: Cho em xin giá phòng để em nhập vào hệ thống CRM
# ✅ #32 [CRAWL_DATA  ] Tào: 51.1% Bình: 50.5% Crawl: 58.6%
#       Text: Anh cho em hỏi thông tin để em training AI chatbot
# ✅ #33 [CRAWL_DATA  ] Tào: 47.8% Bình: 48.8% Crawl: 56.6%
#       Text: Chị gửi em hình ảnh để em viết báo cáo thị trường
# ✅ #34 [CRAWL_DATA  ] Tào: 51.0% Bình: 50.5% Crawl: 56.8%
#       Text: Em đang crawl dữ liệu cho đồ án tốt nghiệp
# ✅ #35 [CRAWL_DATA  ] Tào: 48.6% Bình: 52.0% Crawl: 59.2%
#       Text: Anh cho em xin thông tin để em phân tích đối thủ
# ✅ #36 [CRAWL_DATA  ] Tào: 44.4% Bình: 47.2% Crawl: 51.0%
#       Text: Em cần data để train model machine learning
# ✅ #37 [CRAWL_DATA  ] Tào: 46.3% Bình: 52.3% Crawl: 55.0%
#       Text: Cho em xem giá tham khảo để em viết bài review
# ❌ #38 [BINH_THUONG ] Tào: 51.1% Bình: 57.9% Crawl: 48.6%
#       Text: Em đang làm research về thị trường nhà trọ quận này
# ✅ #39 [CRAWL_DATA  ] Tào: 50.1% Bình: 50.9% Crawl: 58.2%
#       Text: Anh cho em xin thông tin để em nhập database
# ✅ #40 [CRAWL_DATA  ] Tào: 50.2% Bình: 48.1% Crawl: 55.4%
#       Text: Em cần mẫu tin nhắn để xây dựng chatbot tự động
# ❌ #41 [BINH_THUONG ] Tào: 50.0% Bình: 53.9% Crawl: 47.5%
#       Text: Cho em xem ảnh thực tế để em check chất lượng
# ✅ #42 [CRAWL_DATA  ] Tào: 47.3% Bình: 51.2% Crawl: 53.1%
#       Text: Em đang thu thập dữ liệu cho luận văn master
# ✅ #43 [CRAWL_DATA  ] Tào: 50.3% Bình: 53.5% Crawl: 62.0%
#       Text: Anh gửi em bảng giá để em so sánh với đối thủ
# ✅ #44 [CRAWL_DATA  ] Tào: 49.4% Bình: 52.3% Crawl: 57.6%
#       Text: Em cần thông tin để populate vào app của em
# ✅ #45 [CRAWL_DATA  ] Tào: 52.4% Bình: 53.7% Crawl: 60.9%
#       Text: Cho em hỏi giá để em scraping data xây dựng API

# ======================================================================
# 📊 KẾT QUẢ TỔNG HỢP
# ======================================================================
# ✅ Accuracy: 80.00% (36/45)

# 📋 Confusion Matrix:
#                 |   Pred Tào |  Pred Bình | Pred Crawl
# ------------------------------------------------------------
# TAO_LAO         |          8 |          7 |          0
# BINH_THUONG     |          0 |         15 |          0
# CRAWL_DATA      |          0 |          2 |         13

# 📈 Per-class Accuracy:
#   TAO_LAO     : 53.33% (8/15)
#   BINH_THUONG : 100.00% (15/15)
#   CRAWL_DATA  : 86.67% (13/15