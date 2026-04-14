"""
RAG 系统核心模块

协调各组件实现检索增强生成（Retrieval-Augmented Generation）流程。

核心组件：
- DocumentProcessor: 文档解析和分块
- VectorStore: 向量存储和语义搜索
- AIGenerator: AI 响应生成（带 tool calling）
- SessionManager: 对话会话管理
- ToolManager: 工具注册和执行

RAG 流程：
    用户查询 → RAGSystem.query()
           → AIGenerator.generate_response()（带 tools）
           → Claude 决定调用 search_course_content
           → CourseSearchTool.execute() → VectorStore.search()
           → ChromaDB 语义搜索
           → Claude 综合搜索结果生成回答

使用示例：
    rag = RAGSystem(config)
    rag.add_course_folder("../docs")
    answer, sources = rag.query("MCP 是什么?")
"""

from typing import List, Tuple, Optional, Dict
import os
import logging
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool
from models import Course

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG 系统主控制器

    整合文档处理、向量存储、AI 生成和会话管理，
    提供统一的课程问答接口。

    属性：
        config: 配置对象
        document_processor: 文档处理器
        vector_store: 向量存储
        ai_generator: AI 生成器
        session_manager: 会话管理器
        tool_manager: 工具管理器
        search_tool: 课程搜索工具
    """

    def __init__(self, config):
        """
        初始化 RAG 系统

        根据配置创建所有核心组件并建立关联。

        Args:
            config: Config 配置对象
        """
        self.config = config
        logger.debug(f"Initializing RAGSystem: CHUNK_SIZE={config.CHUNK_SIZE}")

        # 初始化核心组件
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )

        self.vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )

        self.ai_generator = AIGenerator(
            config.LLM_API_KEY,
            config.LLM_MODEL,
            config.LLM_BASE_URL
        )
        logger.debug(f"AI Generator: model={config.LLM_MODEL}, base_url={config.LLM_BASE_URL}")

        self.session_manager = SessionManager(config.MAX_HISTORY)

        # 初始化搜索工具
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        logger.debug("Registered CourseSearchTool")

    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        添加单个课程文档

        解析文档并存入向量数据库。

        Args:
            file_path: 课程文档路径

        Returns:
            (Course 对象, 生成的片段数量)
            失败时返回 (None, 0)
        """
        try:
            logger.debug(f"Processing document: {file_path}")

            # 解析文档
            course, course_chunks = self.document_processor.process_course_document(file_path)
            logger.debug(f"Created {len(course_chunks)} chunks for: {course.title}")

            # 存入向量数据库
            self.vector_store.add_course_metadata(course)
            self.vector_store.add_course_content(course_chunks)

            return course, len(course_chunks)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None, 0

    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        批量添加课程文档

        处理文件夹中所有课程文档，支持增量添加。

        Args:
            folder_path: 文档文件夹路径
            clear_existing: 是否清空现有数据后重新导入

        Returns:
            (添加的课程数, 生成的片段总数)
        """
        total_courses = 0
        total_chunks = 0

        logger.debug(f"Adding folder: {folder_path}, clear={clear_existing}")

        # 清空现有数据（可选）
        if clear_existing:
            logger.info("Clearing existing data...")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return 0, 0

        # 获取已存在的课程标题，避免重复添加
        existing_titles = set(self.vector_store.get_existing_course_titles())
        logger.debug(f"Found {len(existing_titles)} existing courses")

        # 处理每个文档文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # 只处理支持的文件类型
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    course, course_chunks = self.document_processor.process_course_document(file_path)

                    if course and course.title not in existing_titles:
                        # 新课程 - 添加到数据库
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        logger.info(f"Added: {course.title} ({len(course_chunks)} chunks)")
                        existing_titles.add(course.title)
                    elif course:
                        logger.debug(f"Skipped existing: {course.title}")

                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")

        logger.debug(f"Folder complete: {total_courses} courses, {total_chunks} chunks")
        return total_courses, total_chunks

    def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        处理用户查询

        使用 tool calling 方式让 AI 自主决定是否搜索课程内容。

        Args:
            query: 用户问题
            session_id: 可选会话 ID（用于对话上下文）

        Returns:
            (AI 回答, 来源列表)
        """
        logger.debug(f"Query: '{query}' (session: {session_id})")

        # 构建提示词
        prompt = f"""Answer this question about course materials: {query}"""

        # 获取对话历史
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # 调用 AI 生成器（带工具）
        logger.debug("Calling AI generator...")
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )

        # 获取搜索来源
        sources = self.tool_manager.get_last_sources()
        logger.debug(f"Sources: {sources}")

        # 清空来源记录
        self.tool_manager.reset_sources()

        # 更新对话历史
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)
            logger.debug(f"Updated session: {session_id}")

        logger.debug(f"Response: '{response[:100]}...' ({len(response)} chars)")
        return response, sources

    def get_course_analytics(self) -> Dict:
        """
        获取课程统计信息

        Returns:
            包含课程总数和标题列表的字典
        """
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles()
        }