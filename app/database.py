import faiss
import numpy as np
import json
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY, OPENAI_BASE_URL

from langchain.schema import Document  # 导入 Document 类

class KnowledgeBase:
    def __init__(self, data_file="data/documents.json"):
        self.data_file = data_file
        self.documents = self.load_documents()
        self.index = self.create_faiss_index()

    def load_documents(self):
        """从 JSON 文件加载文档"""
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("documents", [])  # 确保返回列表，避免 NoneType 错误

    def create_faiss_index(self):
        """创建 FAISS 向量索引"""
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=OPENAI_API_KEY,
            # dashscope_api_base=OPENAI_BASE_URL
        )

        texts = [doc.get("content", "").strip() for doc in self.documents if isinstance(doc.get("content"), str)]

        # 🔴 确保 texts 非空
        if not texts:
            raise ValueError("❌ 文档内容为空，无法创建嵌入向量")

        try:
            # 拆分文本
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            docs = text_splitter.create_documents(texts)  # 这里返回的是 Document 类型的列表

            # 发送给 OpenAI
            texts = [doc.page_content for doc in docs]

                
            vector_store = FAISS.from_texts(texts, embeddings)  

        except Exception as e:
            print("❌ OpenAI 处理失败:", e)
            raise e  # 抛出异常，终止程序

        return vector_store

    def search(self, query, top_k=3):
        """在 FAISS 索引中查找最相关的文档"""
        results = self.index.similarity_search(query, k=top_k)
        return [r.page_content for r in results]

kb = KnowledgeBase()

