
import os, hashlib
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from .utils import load_json_or_jsonl

def _vec_dir(data_path: str, embed_model: str) -> str:
    base = os.path.join("results", "faiss")
    os.makedirs(base, exist_ok=True)
    h = hashlib.sha1((data_path + embed_model).encode()).hexdigest()[:10]
    return os.path.join(base, f"idx_{h}")

def _load_docs(data_path: str) -> List[Document]:
    rows = load_json_or_jsonl(data_path)
    docs = []
    for r in rows:
        if isinstance(r, str):
            docs.append(Document(page_content=r))
        elif "text" in r:
            docs.append(Document(page_content=r["text"], metadata={k: v for k, v in r.items() if k != "text"}))
        elif "context" in r:
            docs.append(Document(page_content=r["context"], metadata={k: v for k, v in r.items() if k != "context"}))
        else:
            docs.append(Document(page_content=str(r)))
    return docs

def build_or_load_vectorstore(data_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    vecdir = _vec_dir(data_path, embed_model)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    if os.path.exists(vecdir):
        return FAISS.load_local(vecdir, embeddings, allow_dangerous_deserialization=True)

    docs = _load_docs(data_path)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(vecdir)
    return vs

def make_retriever(data_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 3):
    vs = build_or_load_vectorstore(data_path, embed_model=embed_model)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # Attach vectorstore for advanced access (e.g., similarity_search_with_score)
    retriever.vectorstore = vs
    return retriever
