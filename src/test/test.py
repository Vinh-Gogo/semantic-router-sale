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
    # --- NHÓM 1: CÂU HỎI "TÀO LAO" / KHÔNG LIÊN QUAN / TRÊU CHỌC (Test độ nhạy của tập tao-lao.jsonl) ---
    "Anh chủ ơi cho em vay 50k mua cơm hộp nha tối trả",
    "Em sale dễ thương quá, tối nay có rảnh đi cà phê với anh không?",
    "Phòng đẹp đấy, nhưng mà anh hết tiền rồi, cho anh ở nợ vài tháng nhé?",
    "Trời mưa to quá, dột hết ướt cả giường rồi em ơi, đền anh đi!",
    "Bên em có bán kem trộn không, dạo này da anh đen quá.",
    "Nay xổ số miền Bắc đánh con gì dễ trúng hả em?",
    "Anh buồn quá, em hát cho anh nghe một bài rồi anh chốt cọc luôn.",
    "Căn này nhìn phong thủy u ám quá, có ma không em?",
    "Anh thuê phòng xong em qua nấu cơm rửa bát cho anh luôn nhé?",
    "Tháng này anh kẹt quá, cho anh gán nợ bằng con chó cưng được không?",
    "Dạo này Bitcoin lên xuống thất thường, em nghĩ anh có nên bắt đáy không?",
    "Phòng ốc gì mà nóng như cái lò, bật điều hòa 16 độ nãy giờ không mát!",
    "Em ăn cơm chưa?",
    "Cho anh mượn tài khoản Netflix xem đỡ buồn tối nay đi em.",
    "Bên em có nhận cầm đồ không? Anh cắm cái xe máy lấy tiền thuê nhà.",
    "Ủa anh nhắn nhầm số, xin lỗi em nhé.",
    "Mai anh dọn đi luôn, không ở nữa, trả cọc lại cho anh ngay!",
    "Hôm qua anh say quá nôn ra sàn, bên em có người qua dọn không?",
    "Em ơi mua giùm anh bao thuốc lá rồi mang lên phòng 302 nhé.",
    "Tối nay đá banh Việt Nam - Thái Lan mấy giờ em nhỉ?",
    "Sao WiFi pass là 12345678 mà anh nhập mãi không được, lừa đảo à?",
    "Nhà tắm bị tắc bồn cầu rồi, trào hết cả ra ngoài cứu anh với!",
    "Anh vừa cãi nhau với người yêu, em tư vấn tâm lý cho anh tí được không?",
    "Chủ nhà khó tính quá, anh ghét không thèm thuê nữa.",
    "Cho anh xin info bạn nữ ở phòng đối diện đi em.",
    "Có ai nhặt được cái ví của anh rơi ở hầm để xe không?",
    "Alo, tổng đài Viettel đúng không ạ?",
    "Em ơi khu này có quán nhậu nào ngon bổ rẻ chỉ anh với.",
    "Mai anh trốn nợ, có người đến tìm thì bảo anh không ở đây nhé.",
    "Phòng này bao ăn bao ở bao luôn cả người yêu không em?",

    # --- NHÓM 2: CÂU HỎI TIÊU CHUẨN / THÔNG TIN CƠ BẢN (Test tập binh-thuong.jsonl) ---
    "Căn hộ này giá thuê một tháng là bao nhiêu vậy em?",
    "Em cho anh xin thêm hình ảnh thật của căn 1 phòng ngủ nhé.",
    "Địa chỉ chính xác của tòa nhà này ở đâu em?",
    "Giá thuê này đã bao gồm phí quản lý và dọn phòng chưa?",
    "Tiền điện nước ở đây tính theo giá nhà nước hay giá dịch vụ?",
    "Bên mình có căn nào full nội thất xách vali vào ở luôn không?",
    "Có chỗ đậu xe ô tô không em, phí gửi xe tháng bao nhiêu?",
    "Khu vực này có hay bị ngập nước vào mùa mưa không em?",
    "Tòa nhà mình có thang máy và bảo vệ 24/7 không?",
    "Anh muốn đi xem phòng thực tế vào chiều nay có được không?",
    "Căn studio diện tích bao nhiêu mét vuông vậy em?",
    "Có được nấu ăn trong phòng không em? Bếp điện hay gas?",
    "Xung quanh đây có siêu thị hay cửa hàng tiện lợi nào gần không?",
    "Em gửi anh bảng báo giá chi tiết các loại phòng bên em đang có nhé.",
    "Căn này có ban công hay cửa sổ lớn đón nắng không?",
    "Phí dịch vụ hàng tháng bên mình gồm những gì vậy em?",
    "Máy giặt dùng chung hay mỗi phòng có một cái riêng?",
    "Bên em có hỗ trợ đăng ký tạm trú tạm vắng cho người nước ngoài không?",
    "Giờ giấc ở đây có tự do không hay có giờ giới nghiêm?",
    "Wifi mỗi tầng một cục phát hay dùng chung cả nhà?",
    "Hành lang và khu vực chung có được vệ sinh thường xuyên không?",
    "Căn 2 phòng ngủ có 2 nhà vệ sinh riêng biệt không em?",
    "Anh thấy trên mạng đăng giá 8 triệu, sao em báo 9 triệu?",
    "Có phòng nào trống có thể dọn vào đầu tháng sau không?",
    "Nếu anh ký hợp đồng dài hạn 1 năm thì có được giảm giá không?",
    "Sân thượng có khu vực phơi đồ hay BBQ không em?",
    "Hệ thống phòng cháy chữa cháy của tòa nhà có đảm bảo không?",
    "Từ đây ra bến xe trung tâm hoặc trạm MRT đi mất bao lâu?",
    "Anh muốn tìm một căn yên tĩnh, không ồn ào tiếng xe cộ.",
    "Trong phòng đã trang bị sẵn tivi và tủ lạnh chưa?",

    # --- NHÓM 3: CÂU HỎI VỀ HỢP ĐỒNG / PHÁP LÝ / ĐIỀU KHOẢN (Test tập crawl-data hoặc binh-thuong chi tiết) ---
    "Hợp đồng thuê bên mình yêu cầu cọc mấy tháng tiền nhà?",
    "Nếu anh chuyển đi trước thời hạn hợp đồng thì có bị mất cọc không?",
    "Điều khoản tăng giá nhà hàng năm được quy định như thế nào trong hợp đồng?",
    "Bên em có xuất hóa đơn đỏ (VAT) cho công ty thuê được không?",
    "Khi bàn giao phòng, hai bên sẽ có biên bản kiểm kê nội thất rõ ràng chứ?",
    "Anh có được phép tự ý thay đổi màu sơn hoặc khoan tường treo tranh không?",
    "Trường hợp thiết bị điện lạnh bị hỏng hóc thì bên nào chịu chi phí sửa chữa?",
    "Tiền cọc sẽ được hoàn trả trong vòng bao nhiêu ngày sau khi thanh lý hợp đồng?",
    "Có phụ phí gì phát sinh thêm ngoài hợp đồng mà anh cần lưu ý không?",
    "Tòa nhà có quy định nghiêm ngặt về việc dẫn bạn bè về chơi qua đêm không?",
    "Khách đến chơi có phải để lại CMND/CCCD ở chốt bảo vệ không?",
    "Anh nuôi 2 con mèo thì tòa nhà có cho phép không? Có tính thêm phí thú cưng không?",
    "Trường hợp bất khả kháng do dịch bệnh phải trả phòng thì giải quyết sao em?",
    "Hợp đồng thuê tối thiểu là 3 tháng hay 6 tháng?",
    "Anh có thể thanh toán tiền thuê nhà bằng thẻ tín dụng được không?",
    "Nếu đóng tiền nhà trễ hạn thì có bị phạt lãi suất không?",
    "Tòa nhà có nội quy chi tiết về việc vứt rác và phân loại rác không?",
    "Anh muốn làm hợp đồng đứng tên 2 người thì có cần thêm giấy tờ gì không?",
    "Bên em có cam kết không tăng giá điện nước trong suốt thời hạn hợp đồng không?",
    "Khi hết hợp đồng mà anh muốn gia hạn thì cần báo trước bao nhiêu ngày?",

    # --- NHÓM 4: PHẢN HỒI / ĐÁNH GIÁ / TÌNH HUỐNG HỖ TRỢ (Test dữ liệu thực tế) ---
    "Hôm qua anh đến xem mà thấy hành lang hơi tối, bên em có định lắp thêm đèn không?",
    "Anh rất ưng phòng, nhưng giá hơi cao so với ngân sách, em xem xin sếp bớt chút đỉnh được không?",
    "Phòng thực tế hơi nhỏ so với ảnh em chụp góc rộng nhỉ.",
    "Anh gửi định vị rồi em ra đón anh ở đầu hẻm được không, đường rắc rối quá.",
    "Cửa sổ phòng 401 chốt bị lỏng, em báo kỹ thuật lên xem giúp anh.",
    "Hôm nay có lịch dọn phòng mà sao chiều rồi anh về vẫn chưa thấy ai dọn vậy em?",
    "Máy lạnh kêu to quá anh không ngủ được, bảo trì lên kiểm tra giùm nhé.",
    "Hóa đơn điện tháng này của anh tăng đột biến, em check lại công tơ giúp anh.",
    "Tòa nhà dạo này có mùi hôi từ cống bốc lên, em báo ban quản lý xử lý gấp.",
    "Có nhà kế bên sửa chữa ồn quá, tòa nhà mình có quy định giờ thi công không?",
    "Cảm ơn em đã hỗ trợ nhiệt tình, đầu giờ chiều mai anh qua ký hợp đồng.",
    "Em ơi anh lỡ làm vỡ cái gương trong nhà tắm rồi, đền bao nhiêu để anh chuyển khoản?",
    "Sáng nay máy giặt chung báo lỗi rò rỉ nước, mọi người không dùng được em ạ.",
    "Bên mình có dịch vụ giặt ủi lấy liền không, anh đang cần gấp.",
    "Anh muốn thuê thêm một chỗ đậu xe máy nữa thì có chỗ không?",
    "Cuối tuần này anh tổ chức sinh nhật nhỏ trên sân thượng có cần đăng ký trước không?",
    "Anh sắp đi công tác 1 tháng, có thể bảo lưu tiền phòng hoặc giảm giá dịch vụ không?",
    "Thang máy số 2 bấm không nhận tầng, kỹ thuật bên em biết chưa?",
    "Anh chuyển khoản tiền nhà tháng này rồi nhé, em kiểm tra và gửi biên lai cho anh.",
    "Phòng ốc ok, dịch vụ tốt, anh sẽ giới thiệu bạn bè qua bên em thuê."
]
    
    for text in test_cases:
        print(f"\n📝 Input: {text}")
        score_0 = calculate_tao_lao_similarity(text, file_path="data/tao-lao.jsonl")
        score_1 = calculate_tao_lao_similarity(text, file_path="data/binh-thuong.jsonl")
        score_2 = calculate_tao_lao_similarity(text, file_path="data/crawl-data.jsonl")
        print(f"Tào lao: {score_0} %")
        print(f"Bình thường: {score_1} %")
        print(f"Crawl data: {score_2} %")