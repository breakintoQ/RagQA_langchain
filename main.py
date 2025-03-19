from fastapi import FastAPI
from app.database import kb
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, OPENAI_BASE_URL

app = FastAPI()

chat_model = ChatOpenAI(
    model_name="qwen_plus",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

@app.get("/query/")
def query_rag(question: str):
    retrieved_texts = kb.search(question)
    context = "\n".join(retrieved_texts)

    response = chat_model([
        SystemMessage(content="你是一个知识库助手，基于提供的内容回答问题。"),
        HumanMessage(content=f"已知信息：{context}\n\n问题：{question}")
    ])

    return {"answer": response.content}
