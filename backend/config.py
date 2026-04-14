"""
配置模块

集中管理 RAG 系统的所有配置参数，从环境变量加载设置。

配置分类：
- 日志级别：LOG_LEVEL
- LLM API：API密钥、模型名称、Base URL
- 文本处理：分块大小、重叠长度
- 向量搜索：嵌入模型、最大结果数
- 数据存储：ChromaDB 路径

使用方式：
    from config import config
    print(config.CHUNK_SIZE)  # 800
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()


@dataclass
class Config:
    """
    系统配置类

    所有配置项都有默认值，可通过环境变量覆盖。
    在 .env 文件中设置相应变量即可修改配置。

    属性说明：
        LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR）
        LLM_API_KEY: 大模型 API 密钥
        LLM_BASE_URL: API 基础 URL（支持第三方兼容端点）
        LLM_MODEL: 使用的模型名称
        EMBEDDING_MODEL: 嵌入模型名称（用于向量编码）
        CHUNK_SIZE: 文本分块大小（字符数）
        CHUNK_OVERLAP: 分块重叠长度（保持上下文连贯）
        MAX_RESULTS: 搜索返回的最大结果数
        MAX_HISTORY: 保留的对话历史消息数
        CHROMA_PATH: ChromaDB 数据库存储路径
    """

    # 日志设置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

    # LLM API 配置（OpenAI 兼容格式）
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")  # 第三方 API 端点
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # 嵌入模型配置
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # 文本处理配置
    CHUNK_SIZE: int = 800        # 每个文本块的字符数
    CHUNK_OVERLAP: int = 100     # 相邻块的重叠字符数
    MAX_RESULTS: int = 5         # 搜索返回的最大结果数
    MAX_HISTORY: int = 2         # 对话历史保留的消息数

    # 数据库路径
    CHROMA_PATH: str = "./chroma_db"


# 全局配置实例
config = Config()