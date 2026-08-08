import fitz  # PyMuPDF
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "all-MiniLM-L6-v2"

class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.chunks = []

    def load_pdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def chunk_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def build_index(self, folder_path="data"):
        all_chunks = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                path = os.path.join(folder_path, file)
                text = self.load_pdf(path)
                chunks = self.chunk_text(text)
                all_chunks.extend(chunks)

        if not all_chunks:
            print("⚠️ No PDFs found in data folder!")
            self.chunks = []
            self.index = None
            return

        self.chunks = all_chunks
        embeddings = self.embedder.encode(all_chunks)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print(f"✅ Indexed {len(all_chunks)} chunks from {folder_path}")

    def retrieve(self, query, top_k=3):
        if self.index is None or not self.chunks:
            return []
        query_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        results = [self.chunks[i] for i in indices[0]]
        return results