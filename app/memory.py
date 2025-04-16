from langchain.schema import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from app.database import get_user_history, save_user_history  # 使用你已有的

class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.store = {}  # 用于存储会话历史

    @property
    def messages(self) -> list[BaseMessage]:
        """
        获取用户的对话历史记录，并将其转换为 BaseMessage 格式。
        """
        raw_history = get_user_history(self.user_id) or []
        message_pairs = []
        for item in raw_history:
            if "问题:" in item and "回答:" in item:
                q = item.split("问题:")[1].split("\n回答:")[0].strip()
                a = item.split("\n回答:")[1].strip()
                message_pairs.append({"type": "human", "data": q})
                message_pairs.append({"type": "ai", "data": a})
        return messages_from_dict(message_pairs)

    def get_session_history(self) -> list[BaseMessage]:
        """
        获取指定会话的历史记录。如果会话不存在，则初始化一个新的会话历史。
        """
        if self.user_id not in self.store:
            self.store[self.user_id] = InMemoryChatMessageHistory()
        return self.store[self.user_id].messages

    def add_message(self, message: BaseMessage) -> None:
        """
        添加消息到用户的历史记录中，并保存到数据库。
        """
        old = get_user_history(self.user_id) or []
        if message.type == "human":
            old.append(f"问题: {message.content}")
        elif message.type == "ai":
            if old and old[-1].startswith("问题:"):
                old[-1] += f"\n回答: {message.content}"
            else:
                old.append(f"回答: {message.content}")
        save_user_history(self.user_id, old)

    def clear(self) -> None:
        """
        清空用户的历史记录。
        """
        save_user_history(self.user_id, [])
        self.store.clear()  # 同时清空内存中的会话历史