import jwt
import datetime
from fastapi import HTTPException
from config import SECRET_KEY

def create_access_token(user_id: str):
    """生成 JWT 访问令牌"""
    payload = {
        "sub": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # 令牌有效期为 1 小时
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str):
    """验证 JWT 令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]  # 返回用户 ID
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def refresh_access_token(token: str):
    """刷新过期的 JWT 令牌"""
    try:
        # 解码令牌，但不验证过期时间
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"], options={"verify_exp": False})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        # 创建新的令牌
        new_token = create_access_token(user_id)
        return {"access_token": new_token, "token_type": "bearer"}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")