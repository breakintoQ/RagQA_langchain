from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.schema import  HumanMessage
from pydantic import BaseModel
from app.database import  get_user_history, save_user_history,get_db
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from app.auth import verify_token, create_access_token
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.knowledgeBase import KnowledgeBase
from app.models import User
from passlib.context import CryptContext
from fastapi import BackgroundTasks
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.memory import CustomChatMessageHistory

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


router = APIRouter()

#创建一个 CryptContext 实例，用于管理密码的哈希和验证。
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class HistoryResponse(BaseModel):
    user_id: str
    history: list

# 请求体模型
class AuthenticatedQueryRequest(BaseModel):
    token: str
    question: str

async def create_faiss_index_background(user_id, file_paths, db: AsyncSession):
    """后台任务：创建 FAISS 索引"""
    async with db.begin():  # 使用上下文管理器
        kb = KnowledgeBase()
        await kb.load_and_create_index(file_paths, user_id, db)

@router.post("/register/")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    # 对用户密码进行哈希处理
    hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, password_hash=hashed_password)
    db.add(new_user)
    await db.commit()
    return {"message": "注册成功"}

@router.post("/login/")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    query = await db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": user.username})
    db_user = query.fetchone()
    if not db_user or not pwd_context.verify(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    token = create_access_token(user_id=str(db_user.id))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(db_user.id),
        "username": db_user.username
    }

@router.post("/query/")
async def query_lcel(request: AuthenticatedQueryRequest, db: AsyncSession = Depends(get_db)):
    try:
        user_id = verify_token(request.token)
        question = request.question

        # 创建知识库索引
        kb = KnowledgeBase()
        if not kb.index:
            await kb.create_faiss_index(user_id, db)
        retrieved_docs = kb.index.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt = ChatPromptTemplate.from_messages(
            [("system", "你是一个知识库助手，会根据历史对话与给定信息回答问题。"),
             MessagesPlaceholder(variable_name="history"),
             ("user", "请根据以下资料回答问题：\n{input}")]
        )
        
        customer_message_history = CustomChatMessageHistory(user_id=user_id)


        config = {"configurable": {"session_id": user_id}}  
        # 3. 构建 LCEL Chain
        chain = prompt | chat_model

        # 4. 执行链，自动记录历史
        answer = await chain.ainvoke(
            {
                "input": f"{context}\n\n问题：{question}",
                "history": customer_message_history.get_session_history(),
            },
            config=config,
        )

        customer_message_history.add_message(HumanMessage(content=question))
        customer_message_history.add_message(answer)
        
        return {"answer": answer.content}
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        raise HTTPException(status_code=500, detail="查询失败")
    

@app.get("/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str):
    """获取用户的对话历史"""
    history = get_user_history(user_id)
    return HistoryResponse(user_id=user_id, history=history)

from typing import List

@app.post("/upload-files/")
async def upload_files(
    background_tasks: BackgroundTasks,
    user_id: str = Query(...),  # 从查询参数获取 user_id
    db: AsyncSession = Depends(get_db),
    files: List[UploadFile] = File(...),  # 接收多个文件
):
    try:
        file_paths = []  # 用于存储所有文件的路径

        # 保存每个文件到临时目录
        for file in files:
            file_path = f"temp_{file.filename}"
            try:
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())
                print(f"文件已保存到: {file_path}")  # 添加调试日志
                file_paths.append(file_path)
            except Exception as e:
                print(f"❌ 保存文件失败: {file.filename}, 错误: {e}")

        if not file_paths:
            raise HTTPException(status_code=400, detail="未上传任何有效文件")

        print(f"用户 {user_id} 上传了文件: {file_paths}")

        # 异步创建索引
        background_tasks.add_task(create_faiss_index_background, user_id, file_paths, db)

        return {"message": "文件上传成功，索引正在后台创建"}
    except Exception as e:
        print(f"❌ 文件上传失败: {e}")
        return {"message": "文件上传失败", "error": str(e)}

app.include_router(router)