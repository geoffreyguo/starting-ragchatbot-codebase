"""
会话管理模块

管理用户对话会话和历史消息记录，支持多轮对话上下文保持。

核心功能：
- 创建/管理会话
- 记录对话历史（用户消息和助手回复）
- 自动限制历史长度，防止内存溢出
- 格式化历史记录供 AI 生成器使用

使用流程：
    manager = SessionManager(max_history=2)
    session_id = manager.create_session()
    manager.add_exchange(session_id, "问题", "回答")
    history = manager.get_conversation_history(session_id)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    消息模型

    表示对话中的单条消息。

    属性：
        role: 消息角色（"user" 或 "assistant"）
        content: 消息内容文本
    """
    role: str
    content: str


class SessionManager:
    """
    会话管理器

    维护多个用户的对话会话，每个会话独立存储消息历史。

    属性：
        max_history: 保留的最大消息数（用户+助手消息总数）
        sessions: 会话字典，键为 session_id，值为消息列表
        session_counter: 会话计数器，用于生成唯一 ID

    使用示例：
        manager = SessionManager(max_history=5)
        sid = manager.create_session()
        manager.add_exchange(sid, "什么是 MCP?", "MCP 是...")
        history = manager.get_conversation_history(sid)
    """

    def __init__(self, max_history: int = 5):
        """初始化会话管理器，设置最大历史消息数"""
        self.max_history = max_history
        self.sessions: Dict[str, List[Message]] = {}
        self.session_counter = 0
        logger.debug(f"SessionManager initialized: max_history={max_history}")

    def create_session(self) -> str:
        """
        创建新会话

        生成唯一 session_id 并初始化空消息列表。

        Returns:
            新会话的 ID 字符串（如 "session_1"）
        """
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        self.sessions[session_id] = []
        logger.debug(f"Created session: {session_id}")
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """
        添加单条消息到会话历史

        自动裁剪超出限制的历史消息，保留最新的消息。

        Args:
            session_id: 会话 ID
            role: 消息角色（"user" 或 "assistant"）
            content: 消息内容
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        message = Message(role=role, content=content)
        self.sessions[session_id].append(message)
        logger.debug(f"Added message to session {session_id}: role={role}, length={len(content)}")

        # 裁剪历史消息，保持不超过 max_history * 2 条
        # 因为每轮对话包含用户消息和助手消息各一条
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]
            logger.debug(f"Trimmed session {session_id} to {len(self.sessions[session_id])} messages")

    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """
        添加完整对话轮次

        一次性添加用户问题和助手回答，简化调用流程。

        Args:
            session_id: 会话 ID
            user_message: 用户输入的问题
            assistant_message: AI 生成的回答
        """
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)
        logger.debug(f"Added exchange to session {session_id}")

    def get_conversation_history(self, session_id: Optional[str]) -> Optional[str]:
        """
        获取格式化的对话历史

        将历史消息格式化为文本字符串，供 AI 作为上下文使用。

        Args:
            session_id: 会话 ID

        Returns:
            格式化的历史文本，如 "User: 问题\\nAssistant: 回答"
            如果会话不存在或无消息，返回 None
        """
        if not session_id or session_id not in self.sessions:
            logger.debug(f"Session not found: {session_id}")
            return None

        messages = self.sessions[session_id]
        if not messages:
            logger.debug(f"No messages in session: {session_id}")
            return None

        # 格式化：每条消息为 "角色: 内容"
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"{msg.role.title()}: {msg.content}")

        history = "\n".join(formatted_messages)
        logger.debug(f"Retrieved history: {len(messages)} messages, {len(history)} chars")
        return history

    def clear_session(self, session_id: str):
        """
        清空会话历史

        删除指定会话的所有消息记录，保留会话 ID。

        Args:
            session_id: 要清空的会话 ID
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []
            logger.debug(f"Cleared session: {session_id}")