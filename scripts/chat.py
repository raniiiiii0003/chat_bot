import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0, "max_length": 512}
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

print("Chatbot ready! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.invoke({"query": query})
    print("Bot:", result["result"])
