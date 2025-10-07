import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def create_vectors():
    # Paths
    DATA_DIR = "./data"
    DB_DIR = "./vector_store"

    # 1. Load documents
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.lower().endswith((".txt", ".md")):
            loader = TextLoader(path, encoding="utf-8")
        else:
            print(f"Skipping unsupported file: {file}")
            continue
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create embeddings
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_DIR)

    print(f"✅ Vector store saved to {DB_DIR}")


if __name__ == "__main__":
    create_vectors()
