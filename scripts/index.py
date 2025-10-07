from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents from "docs" folder
loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create FAISS vector store
db = FAISS.from_documents(documents, embeddings)

# 4. Save index locally
db.save_local("faiss_index")

print("✅ FAISS index created and saved in 'faiss_index/'")
