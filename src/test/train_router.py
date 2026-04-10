import os
import json
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

# Cấu hình file dữ liệu
classes = {
    "TAO_LAO": "data/tao-lao.jsonl",
    "BINH_THUONG": "data/binh-thuong.jsonl",
    "CRAWL_DATA": "data/crawl-data.jsonl"
}

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",
    base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    # base_url="https://api.novita.ai/openai",
    # api_key=SecretStr(os.getenv("NOVITA_API_KEY", "text")),
    check_embedding_ctx_length=False,
    chunk_size=32
    
)

def load_and_embed_data():
    X_vectors = []
    y_labels = []
    
    for label, file_path in classes.items():
        print(f"Đang xử lý dữ liệu lớp: {label}...")
        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        text = record.get("conversation", "").strip()
                        if text:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        
        # Batch embedding cho nhanh
        BATCH_SIZE = 32
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            batch_vecs = embeddings.embed_documents(batch)
            X_vectors.extend(batch_vecs)
            y_labels.extend([label] * len(batch))
            
    return np.array(X_vectors), np.array(y_labels)

if __name__ == "__main__":
    print("🚀 Bắt đầu đọc và nhúng dữ liệu...")
    X, y = load_and_embed_data()
    
    print("🧠 Bắt đầu huấn luyện mô hình Logistic Regression...")
    # class_weight='balanced' giúp xử lý việc các file jsonl có số lượng dòng không đều nhau
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X, y)
    
    print("\n📊 BÁO CÁO ĐỘ CHÍNH XÁC (Trực tiếp trên tập huấn luyện):")
    print(classification_report(y, clf.predict(X)))
    
    # Lưu mô hình lại
    model_path = "src/test/router_model.pkl"
    joblib.dump(clf, model_path)
    print(f"✅ Đã lưu mô hình học máy tại: {model_path}")