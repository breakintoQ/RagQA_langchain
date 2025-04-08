from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from app.database import kb, get_user_history, save_user_history
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from app.auth import verify_token, create_access_token
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import get_db
from app.models import User
from passlib.context import CryptContext

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

@router.post("/register/")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    # hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, password=user.password)
    db.add(new_user)
    await db.commit()
    return {"message": "注册成功"}

@router.post("/login/")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    query = await db.execute(text("SELECT * FROM users WHERE username = :username"), {"username": user.username})
    db_user = query.fetchone()
    if not db_user or db_user.password_hash != user.password:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    token = create_access_token(user_id=str(db_user.id))
    return {
    "access_token": token,
    "token_type": "bearer",
    "user_id": str(db_user.id),
    "username": db_user.username
}

@app.post("/query/")
def query_rag(request: AuthenticatedQueryRequest):
    user_id = verify_token(request.token)  # 验证并获取用户 ID
    question = request.question

    # 获取用户的对话历史
    history = get_user_history(user_id)
    context = "\n".join(history)  # 将历史对话作为上下文

    # 从知识库中检索相关内容
    retrieved_texts = kb.search(question)
    context += "\n" + "\n".join(retrieved_texts)

    # 生成回答
    response = chat_model([
        SystemMessage(content="你是一个知识库助手，基于提供的内容回答问题。"),
        HumanMessage(content=f"已知信息：{context}\n\n问题：{question}")
    ])

    # 更新历史对话
    history.append(f"问题: {question}\n回答: {response.content}")
    save_user_history(user_id, history)

    return {"answer": response.content}

@app.get("/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str):
    """获取用户的对话历史"""
    history = get_user_history(user_id)
    return HistoryResponse(user_id=user_id, history=history)

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """处理文件上传并创建索引"""
    try:
        # 保存文件到临时目录
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 加载文件并创建 FAISS 索引
        kb.load_and_create_index(file_path)
        
        # 返回成功消息
        return JSONResponse(content={"message": "文件上传并处理成功!"}, status_code=200)
    except Exception as e:
        print(f"❌ 文件上传失败: {e}")
        return JSONResponse(content={"message": "文件上传失败", "error": str(e)}, status_code=500)

app.include_router(router)