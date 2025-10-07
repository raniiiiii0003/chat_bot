import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.prompts import PromptTemplate
import gradio as gr

# Paths
DB_DIR = "./vector_store"

# 1. Load embeddings (must match what we used in ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load FAISS vector store
vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Load local model (CPU friendly)
# model_name = "google/flan-t5-large" # Text2Text generation model but week to fetch data
# model_name = "openchat/openchat_3.5" # Big model
model_name = "tiiuae/falcon-7b-instruct"  # Works
llm_pipeline = pipeline("text-generation", model=model_name, device=0, return_full_text=False, max_new_tokens=256,
                        do_sample=True, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

prompt_template = """
    You are a helpful assistant. 
    ONLY use the provided context to answer the question. 
    Do not use prior knowledge. If the context does not contain the answer, reply exactly:
    "I don’t know based on the provided documents."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
   """
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 4. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)


def chatbot_fn(user_input, history):
    if user_input.lower() in ["hi", "hello", "hey"]:
        history.append((user_input, "Hi, how can I help you?"))
        return history

    result = qa.invoke({"query": user_input})
    answer = result["result"]
    history.append((user_input, answer))
    return history


with gr.Blocks() as demo:
    gr.Markdown("## Rani's Chat Bot For College Project")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me about the documents shared...")
    clear = gr.Button("Clear Chat")

    state = gr.State([])


    def respond(message, history):
        history = chatbot_fn(message, history or [])
        return history, history, ""


    msg.submit(respond, [msg, state], [chatbot, state, msg])
    clear.click(lambda: ([], []), None, [chatbot, state], queue=False)

if __name__ == "__main__":
    demo.launch()
