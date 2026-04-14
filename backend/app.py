"""
FastAPI 应用入口

提供 Web 界面和 REST API 接口，是 RAG 系统的 HTTP 服务层。

核心功能：
- 静态文件服务：前端 HTML/JS/CSS
- REST API：
  - POST /api/query：处理用户查询
  - GET /api/courses：获取课程统计
- CORS 支持：允许跨域访问
- 会话管理：自动创建/维护对话会话

API 端点：
    /api/query
        Request: {"query": "问题", "session_id": "可选"}
        Response: {"answer": "回答", "sources": ["来源"], "session_id": "会话ID"}

    /api/courses
        Response: {"total_courses": 5, "course_titles": ["课程1", ...]}

启动方式：
    cd backend && uv run uvicorn app:app --reload --port 8000

访问地址：
    Web 界面: http://localhost:8000
    API 文档: http://localhost:8000/docs
"""

import warnings
# 忽略 resource_tracker 警告
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
from config import config

# 配置日志
log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.DEBUG)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制第三方库的详细日志
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path

from rag_system import RAGSystem

# 创建 FastAPI 应用
app = FastAPI(title="Course Materials RAG System", root_path="")

# 添加信任主机中间件（代理环境）
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# 配置 CORS（跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 初始化 RAG 系统
rag_system = RAGSystem(config)


# API 请求/响应模型

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str                          # 用户问题
    session_id: Optional[str] = None    # 可选会话 ID


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str           # AI 生成的回答
    sources: List[str]    # 引用的来源列表
    session_id: str       # 会话 ID


class CourseStats(BaseModel):
    """课程统计模型"""
    total_courses: int          # 课程总数
    course_titles: List[str]    # 课程标题列表


# API 端点定义

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    处理用户查询

    流程：
    1. 创建或使用现有会话
    2. 调用 RAG 系统处理查询
    3. 返回回答和来源信息

    Args:
        request: QueryRequest 对象

    Returns:
        QueryResponse 对象

    Raises:
        HTTPException: 处理错误时返回 500
    """
    try:
        logger.debug(f"Received query: {request.query}")

        # 创建会话（如果未提供）
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
            logger.debug(f"Created session: {session_id}")

        # 调用 RAG 系统处理查询
        answer, sources = rag_system.query(request.query, session_id)
        logger.debug(f"Response: {answer[:100]}... (sources: {len(sources)})")

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """
    获取课程统计信息

    Returns:
        CourseStats 对象

    Raises:
        HTTPException: 处理错误时返回 500
    """
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件

    自动加载 docs 目录下的课程文档。
    """
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.debug("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.debug(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")


class DevStaticFiles(StaticFiles):
    """
    开发模式静态文件处理器

    为静态文件添加 no-cache 头，确保开发时刷新页面获取最新内容。
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # 添加禁止缓存头
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# 挂载前端静态文件
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")