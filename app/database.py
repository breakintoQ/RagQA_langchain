import json
import redis
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
    """创建一个异步数据库会话，并在请求结束后自动关闭会话"""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()  # 确保会话被关闭

# 连接 Redis 数据库
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

def get_user_history(user_id):
    """从 Redis 获取用户历史对话"""
    history = redis_client.get(user_id)
    return json.loads(history) if history else []

def save_user_history(user_id, history):
    """保存用户对话历史到 Redis"""
    redis_client.set(user_id, json.dumps(history))

