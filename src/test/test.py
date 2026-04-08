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
        # === TÀO LAO (12 samples) ===
        ("Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả", "TAO_LAO"),
        ("Em sale dễ thương quá, tối nay có rảnh đi cà phê với anh không?", "TAO_LAO"),
        ("Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ vài tháng nhé?", "TAO_LAO"),
        ("Bên em có bán kem trộn không, dạo này da anh đen quá.", "TAO_LAO"),
        ("Nay xổ số miền Bắc đánh con gì dễ trúng hả em?", "TAO_LAO"),
        ("Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt cọc luôn.", "TAO_LAO"),
        ("Căn này nhìn phong thủy u ám quá, có ma không em?", "TAO_LAO"), # Có thể châm chước
        ("Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn nhé?", "TAO_LAO"),
        ("Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng được không?", "TAO_LAO"),
        ("Em ăn cơm chưa?", "TAO_LAO"),
        ("Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi em.", "TAO_LAO"),
        ("Phòng này bao ăn bao ở bao luôn cả người yêu không em?", "TAO_LAO"),
        ("Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.", "TAO_LAO"),
        
        # === BÌNH THƯỜNG (18 samples - Thêm các ca xử lý mâu thuẫn/vận hành) ===
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
        ("Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền anh đi!", "BINH_THUONG"), # Đã fix
        ("Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh ngay!", "BINH_THUONG"), # Đã fix
        ("Cho em xem ảnh thực tế để em check chất lượng", "BINH_THUONG"), # Đã fix từ Crawl_Data sang
        
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
    # Test cases: (text, true_label)
    # test_cases = [
    #     # === TÀO LAO (12 samples) ===
    #     # Các câu khách nhắn hoặc spam không liên quan gì đến nghiệp vụ thuê nhà
    #     ("Hôm nay trời mưa to quá, đi đường ngập hết cả xe.", "TAO_LAO"),
    #     ("Bạn ăn cơm chưa hay đang bận chạy KPI?", "TAO_LAO"),
    #     ("Tự nhiên buồn ngủ quá, chả muốn làm gì.", "TAO_LAO"),
    #     ("123456 test bot alo alo", "TAO_LAO"),
    #     ("Cuối tuần này có phim gì ngoài rạp hay không nhỉ?", "TAO_LAO"),
    #     ("Anh đẹp trai thế này có người yêu chưa?", "TAO_LAO"),
    #     ("Bấm nhầm, không có gì đâu nhé.", "TAO_LAO"),
    #     ("Xin lỗi tôi gửi nhầm tin nhắn cho người khác.", "TAO_LAO"),
    #     ("Haha ok ok buồn cười thật đấy =)))", "TAO_LAO"),
    #     ("Cho mình vay 5 triệu qua tuần trả được không?", "TAO_LAO"),
    #     ("Hôm qua xem đá bóng MU thua chán ghê.", "TAO_LAO"),
    #     ("gửi ảnh con mèo qua xem nào", "TAO_LAO"),
        
    #     # === BÌNH THƯỜNG (18 samples - Nghiệp vụ Sale BĐS cho thuê/Nhắn tin khách hàng) ===
    #     # Tập trung vào hỏi đáp giá, tiện ích, cọc, hợp đồng, lịch xem nhà
    #     ("Anh muốn xem hợp đồng mẫu trước khi quyết định thuê", "BINH_THUONG"),
    #     ("Căn 2 ngủ ở Masteri dạo này giá thuê tầm bao nhiêu em?", "BINH_THUONG"),
    #     ("Giá 15 triệu này là đã bao gồm phí quản lý tòa nhà chưa bạn?", "BINH_THUONG"),
    #     ("Tối nay khoảng 7h chị ghé xem thực tế căn góc tầng 12A được không?", "BINH_THUONG"),
    #     ("Em gửi anh video quay cận cảnh phòng khách và toilet căn The Sun Avenue nha.", "BINH_THUONG"),
    #     ("Chủ nhà có bớt chút lộc không em, anh cọc luôn và dọn vào ở đầu tháng sau.", "BINH_THUONG"),
    #     ("Căn này hướng ban công là gì thế? Buổi chiều có bị hắt nắng gắt không?", "BINH_THUONG"),
    #     ("Khu này có gym với hồ bơi free cho cư dân không em, hay phải đóng phí riêng?", "BINH_THUONG"),
    #     ("Bên mình ký hợp đồng tối thiểu 1 năm hay 6 tháng cũng được hả em?", "BINH_THUONG"),
    #     ("Nếu anh thuê dài hạn thì chủ nhà có hỗ trợ sắm thêm cho cái máy giặt không?", "BINH_THUONG"),
    #     ("Mai mấy giờ anh rảnh, em dẫn anh lên check lại nội thất bàn giao trước khi chốt nhé.", "BINH_THUONG"),
    #     ("Anh vừa chuyển khoản tiền cọc 10 triệu rồi, em check tin nhắn báo chủ nhà giữ căn giúp anh.", "BINH_THUONG"),
    #     ("Phí gửi xe máy và ô tô dưới hầm chung cư này mỗi tháng là bao nhiêu vậy?", "BINH_THUONG"),
    #     ("Cho chị hỏi nếu đơn phương chấm dứt hợp đồng trước hạn 2 tháng thì có mất cọc không?", "BINH_THUONG"),
    #     ("Dạ căn 3 phòng ngủ ở Landmark 81 đang trống, đầy đủ nội thất xách vali vào ở liền được luôn ạ.", "BINH_THUONG"),
    #     ("Khách cũ vừa dọn đi nên thứ 3 tuần sau mới có người dọn vệ sinh công nghiệp xong nha anh.", "BINH_THUONG"),
    #     ("Chung cư này ban quản lý có cho nuôi thú cưng không bạn, mình có một bé poodle nhỏ.", "BINH_THUONG"),
    #     ("Căn hộ 1PN+1 bên này hỗ trợ cho người nước ngoài đăng ký tạm trú không em?", "BINH_THUONG"),
        
    #     # === CRAWL_DATA (14 samples - Lệnh cào dữ liệu, dễ bị nhầm với "lấy/xem thông tin") ===
    #     ("Viết script Python để crawl toàn bộ giá thuê căn hộ trên Batdongsan.com.vn khu vực Quận 2.", "CRAWL_DATA"),
    #     ("Lấy tất cả bình luận và ID người dùng của bài viết môi giới này trên fanpage Facebook.", "CRAWL_DATA"),
    #     ("Thu thập thông tin liên hệ của chính chủ cho thuê nhà trên trang Chợ Tốt.", "CRAWL_DATA"),
    #     ("Cào dữ liệu lịch sử biến động giá cho thuê chung cư Vinhomes Central Park trong vòng 3 năm qua.", "CRAWL_DATA"),
    #     ("Scrape danh sách dự án, giá phòng và đánh giá của cư dân trên diễn đàn Otosaigon.", "CRAWL_DATA"),
    #     ("Trích xuất tự động tiêu đề, tóm tắt và đường link của các bài đăng cho thuê nhà mới nhất.", "CRAWL_DATA"),
    #     ("Xây dựng tool crawl data danh sách căn hộ kèm diện tích và số phòng ngủ trên Rever.", "CRAWL_DATA"),
    #     ("Lấy dữ liệu text và hình ảnh từ các bài post trên group Hội Cư Dân chung cư X.", "CRAWL_DATA"),
    #     ("Viết tool tự động duyệt qua các trang và tải về tất cả ảnh mặt bằng độ phân giải cao từ website dự án.", "CRAWL_DATA"),
    #     ("Thu thập thông tin giá thuê, năm bàn giao, và phí quản lý của các chung cư cũ trên trang Propzy.", "CRAWL_DATA"),
    #     ("Crawl toàn bộ review 1 sao của khách hàng về ban quản lý tòa nhà này trên Google Maps.", "CRAWL_DATA"),
    #     ("Cào nội dung số điện thoại từ các bình luận trên Tiktok của kênh review bất động sản.", "CRAWL_DATA"),
    #     ("Thiết lập bot chạy cronjob hằng ngày để lấy data căn hộ đăng mới trên website.", "CRAWL_DATA"),
    #     ("Scraping dữ liệu danh sách khách hàng từ phần mềm CRM xuất ra file excel tự động.", "CRAWL_DATA")
    # ]
    
    # Đánh giá
    correct = 0
    total = len(test_cases)
    
    # Confusion matrix: {true: {pred: count}}
    labels = ["TAO_LAO", "BINH_THUONG", "CRAWL_DATA"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    
    print(f"\n🔍 Bắt đầu test {total} samples...\n")
    
    for i, (text, true_label) in enumerate(test_cases, 1):
        pred_label, scores = predict_3_class("Customer: " + text)
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        confusion[true_label][pred_label] += 1
        
        status = "✅" if is_correct else "❌"
        print(f"  Text: {text}")
        print(f"{status} #{i:02d} [{pred_label:12s}] Tào:{scores['TAO_LAO']:5.1f}% "
              f"Bình:{scores['BINH_THUONG']:5.1f}% Crawl:{scores['CRAWL_DATA']:5.1f}%")
    
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