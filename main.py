from fastapi import FastAPI
from app.database import kb
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, OPENAI_BASE_URL

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域访问，可以限制为特定域，如 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（GET, POST等）
    allow_headers=["*"],  # 允许所有请求头
)

chat_model = ChatOpenAI(
    model_name="qwen-plus",
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
