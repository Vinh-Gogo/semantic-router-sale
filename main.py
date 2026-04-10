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


# Đừng chuẩn hóa ngay, hãy lấy vector thô
raw_vec_A = embeddings.embed_query("cứt")
raw_vec_B = embeddings.embed_query("danh")

# Tính độ dài (L2 Norm)
mag_A = np.linalg.norm(raw_vec_A)
mag_B = np.linalg.norm(raw_vec_B)

print(f"Độ dài A: {mag_A}")
print(f"Độ dài B: {mag_B}")

cos_sim = np.dot(raw_vec_A, raw_vec_B) / (mag_A * mag_B)
print(cos_sim)