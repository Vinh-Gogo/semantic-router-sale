import os
import sys
import numpy as np
import requests
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from vector_store import normalize_vnese, normalize_record
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

# Qdrant init
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def preprocess_text(text: str) -> str:
    """Chuẩn hóa text trước khi đưa qua mô hình Embedding"""
    if not text or not str(text).strip():
        return ""
    cleaned = str(normalize_record(text)).lower()
    return normalize_vnese(cleaned)

def read_texts_from_jsonl(dataset_name: str) -> list[str]:
    """Đọc dữ liệu văn bản trực tiếp từ file jsonl"""
    file_path = f"data/{dataset_name}.jsonl"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if 'message' in record and record['message'].strip():
                        data.append(record['message'].strip())
                except json.JSONDecodeError:
                    continue
    return data

def init_qdrant_collection(dataset_name: str) -> QdrantVectorStore:
    """Khởi tạo Qdrant collection, tự động import dữ liệu nếu collection trống"""
    
    collections = qdrant_client.get_collections()
    exists = any(c.name == dataset_name for c in collections.collections)
    
    if not exists:
        print(f"🔄 Đang tạo mới và import [{dataset_name}] vào Qdrant...")
        raw_messages = read_texts_from_jsonl(dataset_name)
        messages = [preprocess_text(msg) for msg in raw_messages if preprocess_text(msg)]
        
        # Dùng from_texts để LangChain tự đo dimension và tạo collection
        vectorstore = QdrantVectorStore.from_texts(
            texts=messages,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=dataset_name,
        )
        print(f"✅ Đã import {len(messages)} document vào Qdrant collection: {dataset_name}")
        return vectorstore
    
    # Nếu collection đã tồn tại, chỉ cần kết nối lại
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=dataset_name,
        embedding=embeddings
    )

# Khởi tạo 3 vector store 1 lần duy nhất
vector_stores = {
    "tao-lao": init_qdrant_collection("tao-lao"),
    "binh-thuong": init_qdrant_collection("binh-thuong"),
    "crawl-data": init_qdrant_collection("crawl-data")
}

async def predict_3_class(input_text: str, k: int = 3):
    """Dự đoán lớp dùng Qdrant native similarity search"""
    processed_text = preprocess_text(input_text)
    
    if not processed_text:
        return "BINH_THUONG", {"TAO_LAO": 0, "BINH_THUONG": 50, "CRAWL_DATA": 0}
    
    scores = {}
    
    # Search trên từng collection riêng biệt
    for label, vs in vector_stores.items():
        results = await vs.asimilarity_search_with_score(query=processed_text, k=k)
        
        # Lấy score cao nhất trong K kết quả gần nhất
        max_score = 0.0
        for _, score in results:
            if score > max_score:
                max_score = score
        
        # Convert sang % giống format cũ
        display_label = label.replace("-", "_").upper()
        scores[display_label] = float(max_score) * 100
    
    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores

import asyncio

async def main():
    print("🚀 Đánh giá Accuracy 3 lớp (Qdrant Native Retrieval)")
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
    
    correct = 0
    total = len(test_cases)
    
    labels = ["TAO_LAO", "BINH_THUONG", "CRAWL_DATA"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    
    print(f"\n🔍 Bắt đầu test {total} samples...\n")
    
    for i, (text, true_label) in enumerate(test_cases, 1):
        pred_label, scores = await predict_3_class(text)
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        confusion[true_label][pred_label] += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} #{i:02d} [{pred_label:12s}] Tào:{scores['TAO_LAO']:5.1f}% "
              f"Bình:{scores['BINH_THUONG']:5.1f}% Crawl:{scores['CRAWL_DATA']:5.1f}%")
        print(f"      Text: {text[:55]}{'...' if len(text) > 55 else ''}")
    
    accuracy = correct / total * 100
    
    print(f"\n{'='*70}")
    print(f"📊 KẾT QUẢ TỔNG HỢP")
    print(f"{'='*70}")
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    asyncio.run(main())
