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
        """Legacy method to load entire PDF text without page numbers."""
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def chunk_text(self, text):
        """Legacy method to chunk text without page numbers."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def process_pdf(self, path):
        doc = fitz.open(path)
        filename = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks_with_meta = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            page_chunks = splitter.split_text(text)
            for chunk in page_chunks:
                chunks_with_meta.append(f"Source: {filename}, Page: {page_num}\n{chunk}")
        return chunks_with_meta

    def build_index(self, folder_path="data"):
        all_chunks = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                path = os.path.join(folder_path, file)
                chunks = self.process_pdf(path)
                all_chunks.extend(chunks)

        self.chunks = all_chunks
        embeddings = self.embedder.encode(all_chunks)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print(f"✅ Indexed {len(all_chunks)} chunks from {folder_path}")

    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        results = [self.chunks[i] for i in indices[0]]
        return results