import os
import re
import pandas as pd

import numpy as np
# import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SQLiteVec
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

import unicodedata
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

TONE_NORM_VI = {
    'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
    'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
    'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
    'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
    'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
    'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
    'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
    'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
    'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
    'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
    'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
    'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
    'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
    'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
    'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
    }

def normalize_vnese(text):
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Remove control characters (ASCII 0–31, plus DEL 127)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    return text

# -------------------------
# Load CSV
# -------------------------
def add_column_names_to_values(df : pd.DataFrame) -> pd.DataFrame:
    # Đọc file csv
    
    # Áp dụng cho từng hàng
    def row_with_keys(row):
        return {col: f"{col}: {row[col]}" for col in df.columns}
    
    # Tạo DataFrame mới với giá trị đã thêm tên cột
    new_df = df.apply(row_with_keys, axis=1, result_type="expand")
    return new_df

def load_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    # Normalize current_price
    df["current_price"] = (
        df["current_price"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
    )
    df["current_price"] = df["current_price"].astype(float).dropna().astype("Int64").astype(str) + " vnd"
    
    # df = add_column_names_to_values(df)
    
    return df


def convert_table_to_rows(df: pd.DataFrame) -> list[str]:
    result_list = []
    for _, row in df.iterrows():
        row_string = ", ".join(str(v) for v in row)
        result_list.append(re.sub(r'<[^>]*>|\s+', ' ', row_string).strip())
    return result_list

# -------------------------
# Text preprocessing
# -------------------------
def remove_stopwords_vi(text: str, path_documents_vi: str='stopwords-vietnamese.txt') -> str:
    if not os.path.exists(path_documents_vi):
        raise FileNotFoundError(f"Stopwords file not found: {path_documents_vi}")
    
    tokens = text.split(',')
    id_str, link_str, name_str = tokens[:3]
    content = ','.join(tokens[3:])
    
    stop_words = set(open(path_documents_vi, encoding="utf-8").read().splitlines())
    filtered_tokens = [w.strip() for w in word_tokenize(content, format="text").split(',') if w.strip().lower() not in stop_words]
    
    return f"{id_str},{link_str},{name_str}," + ', '.join(filtered_tokens)

def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    tokens = text.split(',')
    cleaned_tokens = tokens[:3] + [re.sub(r'[^0-9a-zA-ZÀ-Ỹà-ỹ\s]', '', t) for t in tokens[3:]]
    return ' '.join(cleaned_tokens)

def normalize_record(text: str, fix_inch_heu=False) -> str:
    if not text:
        return text
    text = re.sub(r'<.*?>', ' ', text).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_load_data_to_embedding(path: str) -> list:
    df = load_excel(path)
    return convert_table_to_rows(df)

def run_normalization_data(sequences: list, path_stopwords: str="src/stopwords-vietnamese.txt") -> list:
    
    # sequences = [remove_stopwords_vi(seq, path_stopwords) for seq in sequences]
    # sequences = [clean_text(seq) for seq in sequences]
    sequences = [str(normalize_record(seq)).lower() for seq in sequences]
    sequences = [normalize_vnese(seq) for seq in sequences]
    return sequences

def get_model_qwen(device: str = "cuda:0") -> HuggingFaceEmbeddings:
    
    '''For Local'''

    # Load model embedding (phải giống model lúc insert để đảm bảo tương thích vector dim)
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs = {'device': device}
                )

from huggingface_hub import InferenceClient

def get_qwen_embedding_hf_endpoint(base_url: str = 'http://localhost:8080') -> HuggingFaceEndpointEmbeddings:

    ''' For Docker '''

    return HuggingFaceEndpointEmbeddings(
    model=base_url,
    task="feature-extraction",
)

def get_openai_embedding_base_url(base_url: str = 'http://localhost:8080') -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=base_url,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY_EMBED"),
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        dimensions=1024
    )

# -------------------------
# Init vector store
# -------------------------

def init_vectorstore_faiss(model, db_folder: str, action: str ='write'):

    if action == "write":
        docs = run_load_data_to_embedding('./src/store/comque_new.csv')
        docs = run_normalization_data(docs, path_stopwords='./src/store/stopwords-vietnamese.txt')
        docs = [' | '.join(doc.split(', ')) for doc in docs]
        
        dim = len(model.embed_query("hello world"))  # dimension
        index = faiss.IndexFlatL2(dim)

        vector_store = FAISS(
            embedding_function=model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        docs_faiss = [Document(page_content=txt) for txt in docs]
        uuids = [str(uuid4()) for _ in range(len(docs_faiss))]

        # -------------------------------
        # 5. Add to FAISS
        # -------------------------------
        vector_store.add_documents(documents=docs_faiss, ids=uuids)
        
        # SAVE DATABASE
        vector_store.save_local(db_folder)
    else:
        # load FAISS từ ./data
        vector_store = FAISS.load_local(
            folder_path=db_folder,
            embeddings=model,
            allow_dangerous_deserialization=True  # BẬT lên nếu file do bạn tạo
        )
    
    return vector_store

def init_vectorstore(model, db_folder: str, connection, action:str='write') -> SQLiteVec:

    if action == 'write':
        os.makedirs(db_folder, exist_ok=True)

        db_file = db_folder+"/vec.db"
        vt = SQLiteVec(table="state_union", connection=connection, embedding=model)
        
        # Nếu DB chưa tồn tại, thêm documents
        if not os.path.exists(db_file):
            
            docs = run_load_data_to_embedding('../store/comque_new.csv')
            docs = run_normalization_data(docs, path_stopwords='../store/stopwords-vietnamese.txt')
            list_docs = [Document(page_content=dox, metadata={dox.split(',')[0]}) for dox in docs]
            vt.add_documents(list_docs)
        
    return vt
