import os
import json
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import Document
from app.database import redis_client
import asyncio

class KnowledgeBase:
    """
    知识库管理类，支持从多个 JSON 和 TXT 文件加载文档、存储到数据库，并创建 FAISS 索引。
    """

    def __init__(self):
        self.index = None

    def cache_user_documents(self, user_id, documents):
        """将用户文档缓存到 Redis"""
        redis_client.set(f"user_documents:{user_id}", json.dumps(documents))

    def get_cached_user_documents(self, user_id):
        """从 Redis 缓存获取用户文档"""
        cached = redis_client.get(f"user_documents:{user_id}")
        return json.loads(cached) if cached else None
    
    async def load_documents_with_cache(self, user_id, db: AsyncSession):
        """优先从缓存加载文档，如果缓存不存在则从数据库加载"""
        cached_documents = self.get_cached_user_documents(user_id)
        if cached_documents:
            return cached_documents

        # 从数据库加载并缓存
        documents = await self.load_documents_from_db(user_id, db)
        self.cache_user_documents(user_id, documents)
        return documents

    async def save_documents_to_db(self, documents, user_id, db: AsyncSession):
        """将文档保存到数据库"""
        try:
            for doc in documents:
                if not doc.get("content"):  # 检查文档内容是否为空
                    print(f"跳过空文档: {doc}")
                    continue
                new_doc = Document(
                    user_id=user_id,
                    content=doc.get("content", ""),
                    file_name=doc.get("file_name", None)
                )
                db.add(new_doc)
            await db.commit()
            print(f"成功保存 {len(documents)} 条文档到数据库")
        except Exception as e:
            print("❌ 保存文档到数据库失败:", e)
            await db.rollback()

    async def load_documents_from_db(self, user_id, db: AsyncSession, limit=None, offset=None):
        """从数据库分页加载文档"""
        try:
            query = select(Document).where(Document.user_id == user_id)
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)

            result = await db.execute(query)
            documents = result.scalars().all()
            return [{"content": doc.content, "file_name": doc.file_name} for doc in documents]
        except Exception as e:
            print("❌ 从数据库加载文档失败:", e)
            return []

    async def create_faiss_index(self, user_id, db: AsyncSession):
        """从数据库分批加载文档并创建 FAISS 索引"""
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=OPENAI_API_KEY,
        )

        batch_size = 100
        offset = 0
        all_texts = []

        while True:
            documents = await self.load_documents_from_db(user_id, db, limit=batch_size, offset=offset)
            if not documents:
                break
            texts = [doc["content"].strip() for doc in documents if isinstance(doc["content"], str)]
            all_texts.extend(texts)
            offset += batch_size

        if not all_texts:
            raise ValueError("❌ 文档内容为空，无法创建嵌入向量")

        # 拆分文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = text_splitter.create_documents(all_texts)

        # 创建 FAISS 索引
        texts = [doc.page_content for doc in docs]
        self.index = FAISS.from_texts(texts, embeddings)

    async def load_and_create_index(self, file_paths, user_id, db: AsyncSession):
        """
        根据文件类型加载多个文件，将内容存储到数据库，并创建 FAISS 索引
        :param file_paths: 文件路径列表
        :param user_id: 用户 ID
        :param db: 数据库会话
        """
        documents = []

        # 加载文件内容
        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.json':
                documents.extend(self.load_documents_from_json(file_path))
            elif file_extension == '.txt':
                documents.extend(self.load_documents_from_txt(file_path))
            else:
                print(f"❌ 不支持的文件格式：{file_path}，仅支持 .json 或 .txt 文件")

        if not documents:
            print("❌ 没有加载到任何文档")
            return

        # 检查用户是否已有文档
        existing_documents = await self.load_documents_from_db(user_id, db)
        if not existing_documents:
            print(f"用户 {user_id} 的知识库为空，正在初始化...")
        else:
            print(f"用户 {user_id} 的知识库已有 {len(existing_documents)} 条文档，正在追加新文档...")

        await self.save_documents_to_db(documents, user_id, db)

        # 创建索引
        await self.create_faiss_index(user_id, db)


    def load_documents_from_json(self, data_file):
            """从 JSON 文件加载文档"""
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [{"content": doc.get("content", ""), "file_name": os.path.basename(data_file)} for doc in data.get("documents", [])]
            except Exception as e:
                print("❌ 加载 JSON 文档失败:", e)
                return []

    def load_documents_from_txt(self, data_file):
        """从 TXT 文件加载文档"""
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [{"content": line.strip(), "file_name": os.path.basename(data_file)} for line in lines]
        except Exception as e:
            print("❌ 加载 TXT 文档失败:", e)
            return []
