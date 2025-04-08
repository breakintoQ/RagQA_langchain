import os
import json
import faiss
import numpy as np
import redis
from fastapi import UploadFile, File
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain.schema import Document
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "mysql+aiomysql://root:root@localhost:3306/rag_db"

#用于创建异步数据库引擎
engine = create_async_engine(DATABASE_URL, echo=True)
#用于创建数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
#这是 SQLAlchemy 提供的一个基类，用于定义 ORM 模型（即数据库表的类）
Base = declarative_base()

async def get_db():
    #创建一个异步数据库会话，并在请求结束后自动关闭会话
    async with SessionLocal() as session:
        #将会话对象返回给调用者（例如路由函数），以便在请求中使用。
        yield session


# 连接 Redis 数据库
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

def get_user_history(user_id):
    """从 Redis 获取用户历史对话"""
    history = redis_client.get(user_id)
    return json.loads(history) if history else []

def save_user_history(user_id, history):
    """保存用户对话历史到 Redis"""
    redis_client.set(user_id, json.dumps(history))

class KnowledgeBase:
    """
    知识库管理类，支持从 JSON 和 TXT 文件加载文档并创建 FAISS 索引。
    """

    def __init__(self):
        self.documents = []
        self.index = None

    def load_documents_from_json(self, data_file):
        """从 JSON 文件加载文档"""
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
        except Exception as e:
            print("❌ 加载 JSON 文档失败:", e)

    def load_documents_from_txt(self, data_file):
        """从 TXT 文件加载文档"""
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.documents = [{"content": line.strip()} for line in lines]
        except Exception as e:
            print("❌ 加载 TXT 文档失败:", e)

    def create_faiss_index(self):
        """创建 FAISS 向量索引"""
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=OPENAI_API_KEY,
            # dashscope_api_base=OPENAI_BASE_URL
        )

        texts = [doc.get("content", "").strip() for doc in self.documents if isinstance(doc.get("content"), str)]

        if not texts:
            raise ValueError("❌ 文档内容为空，无法创建嵌入向量")

        try:
            # 拆分文本
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            docs = text_splitter.create_documents(texts)

            # 发送给 OpenAI
            texts = [doc.page_content for doc in docs]
            vector_store = FAISS.from_texts(texts, embeddings)
        except Exception as e:
            print("❌ 创建 FAISS 向量索引失败:", e)
            raise e

        return vector_store

    def search(self, query, top_k=3):
        """在 FAISS 索引中查找最相关的文档"""
        results = self.index.similarity_search(query, k=top_k)
        return [r.page_content for r in results]

    def load_and_create_index(self, file_path):
        """根据文件类型加载文件并创建 FAISS 索引"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.json':
            self.load_documents_from_json(file_path)
        elif file_extension == '.txt':
            self.load_documents_from_txt(file_path)
        else:
            raise ValueError("❌ 不支持的文件格式，仅支持 .json 或 .txt 文件")
        
        self.index = self.create_faiss_index()

# 创建一个知识库实例
kb = KnowledgeBase()