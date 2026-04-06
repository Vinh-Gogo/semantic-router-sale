# import torch
import sys
import os
from pathlib import Path

# Fix for Windows Triton issue
# Unconditionally mock triton on Windows because the installed version is broken/incompatible
if os.name == 'nt':
    sys.modules["triton"] = None

from sentence_transformers import SentenceTransformer
import numpy as np
from src.models.helpers import cosine_similarity

class QwenEmbedding:
    """Lớp để tạo embeddings sử dụng Qwen/Qwen3-Embedding-0.6B multilingual model"""
    
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        print(f"Loading {model_name} model...")
        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            self.device = torch.device("cuda:0")
            print(f"Setting device to: {self.device}")
        else:
            print("CUDA is NOT available. Using CPU.")
            self.device = torch.device("cpu")
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "eager"} if torch.cuda.is_available() else {"attn_implementation": "eager"}
        )
        print(f"Model {model_name} loaded on {self.device} with bfloat16")
    
    def get_embedding(self, text):
        """Tạo embedding cho văn bản"""
        if not text or len(text.strip()) == 0:
            return None
        
        # Cắt text nếu quá dài (giới hạn 512 tokens)
        if len(text) > 2000:
            text = text[:2000]
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16):
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embedding_array(self, texts):
        """Tạo ma trận embedding cho danh sách văn bản"""
        if not texts or len(texts) == 0:
            return np.array([])
        
        # Cắt text nếu quá dài (giới hạn 512 tokens)
        processed_texts = []
        for text in texts:
            if len(text) > 2000:
                processed_texts.append(text[:2000])
            else:
                processed_texts.append(text)
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16):
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, text1, text2) -> float:
        """Tính độ tương đồng giữa 2 văn bản"""
        if not text1 or not text2:
            return 0

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0

        return cosine_similarity(emb1, emb2)

    def similarity_matrix(self, query_emb, doc_embs):
        """Tính ma trận độ tương đồng cosine giữa query và danh sách docs"""
        # Chuyển sang tensor trên device
        query_tensor = torch.tensor(query_emb, dtype=torch.float32).to(self.device)
        doc_tensors = torch.tensor(doc_embs, dtype=torch.float32).to(self.device)

        # Normalize vectors
        query_norm = query_tensor / torch.norm(query_tensor)
        doc_norms = doc_tensors / torch.norm(doc_tensors, dim=1, keepdim=True)

        # Compute cosine similarities
        similarities = torch.matmul(doc_norms, query_norm)

        return similarities
    
    def rank_documents(self, query, documents):
        """
        Xếp hạng documents theo độ liên quan với query
        
        Args:
            query (str): Câu truy vấn
            documents (list[str]): Danh sách documents
            
        Returns:
            tuple: (sorted_docs, sorted_scores, sorted_indices)
                - sorted_docs: Documents đã sắp xếp theo độ liên quan
                - sorted_scores: Điểm tương đồng tương ứng
                - sorted_indices: Chỉ số gốc của documents
        """
        if not query or not documents:
            return [], [], []
        
        # Encode query và documents
        query_embedding = self.get_embedding(query)
        doc_embeddings = self.get_embedding_array(documents)
        
        if query_embedding is None or len(doc_embeddings) == 0:
            return [], [], []
        
        # Tính similarity
        similarities = self.similarity_matrix(query_embedding, doc_embeddings).flatten()

        # Sort documents by cosine similarity (on device)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        similarities_np = similarities.cpu().numpy()
        sorted_docs = [documents[idx] for idx in sorted_indices]
        sorted_scores = [similarities_np[idx] for idx in sorted_indices]
        
        return sorted_docs, sorted_scores, sorted_indices
    
    def find_most_similar(self, query, documents, top_k=5):
        """
        Tìm top-k documents tương tự nhất với query
        
        Args:
            query (str): Câu truy vấn
            documents (list[str]): Danh sách documents
            top_k (int): Số lượng kết quả trả về
            
        Returns:
            list[tuple]: Danh sách (document, score, index) được sắp xếp
        """
        sorted_docs, sorted_scores, sorted_indices = self.rank_documents(query, documents)
        
        # Lấy top-k kết quả
        top_k = min(top_k, len(sorted_docs))
        results = [
            (sorted_docs[i], float(sorted_scores[i]), int(sorted_indices[i]))
            for i in range(top_k)
        ]
        
        return results
