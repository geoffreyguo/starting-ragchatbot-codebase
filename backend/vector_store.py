"""
向量存储模块

基于 ChromaDB 实现课程内容的向量存储和语义搜索。

核心功能：
- 管理两个 ChromaDB 集合：course_catalog（课程元数据）和 course_content（内容片段）
- 使用 SentenceTransformer 进行文本嵌入编码
- 支持语义搜索和元数据过滤（按课程名称、课时编号）
- 智能课程名称匹配（支持部分匹配）

数据结构：
    course_catalog 集合：
        - 存储课程标题、链接、讲师、课时列表
        - 用于课程名称的语义匹配

    course_content 集合：
        - 存储分块后的课程内容
        - 用于实际的内容搜索

使用示例：
    store = VectorStore("./chroma_db", "all-MiniLM-L6-v2")
    results = store.search("MCP 是什么", course_name="MCP", lesson_number=1)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
from models import Course, CourseChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """
    搜索结果容器

    封装 ChromaDB 查询结果，提供统一的数据访问接口。

    属性：
        documents: 搜索到的文档内容列表
        metadata: 文档元数据列表（课程标题、课时编号等）
        distances: 向量距离列表（越小越相似）
        error: 错误信息（搜索失败时）
    """
    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    @classmethod
    def from_chroma(cls, chroma_results: Dict) -> 'SearchResults':
        """
        从 ChromaDB 查询结果创建 SearchResults

        ChromaDB 返回嵌套列表格式，需要提取第一层。

        Args:
            chroma_results: ChromaDB query() 返回的结果字典

        Returns:
            SearchResults 实例
        """
        return cls(
            documents=chroma_results['documents'][0] if chroma_results['documents'] else [],
            metadata=chroma_results['metadatas'][0] if chroma_results['metadatas'] else [],
            distances=chroma_results['distances'][0] if chroma_results['distances'] else []
        )

    @classmethod
    def empty(cls, error_msg: str) -> 'SearchResults':
        """创建带错误信息的空结果"""
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        """检查结果是否为空"""
        return len(self.documents) == 0


class VectorStore:
    """
    向量存储管理器

    使用 ChromaDB 存储课程数据，支持语义搜索和过滤。

    属性：
        client: ChromaDB 客户端实例
        embedding_function: 嵌入函数（用于文本向量化）
        course_catalog: 课程元数据集合
        course_content: 课程内容片段集合
        max_results: 默认最大返回结果数
    """

    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5):
        """
        初始化向量存储

        Args:
            chroma_path: ChromaDB 数据存储路径
            embedding_model: 嵌入模型名称（如 "all-MiniLM-L6-v2"）
            max_results: 默认搜索返回的最大结果数
        """
        self.max_results = max_results
        logger.debug(f"Initializing VectorStore: path={chroma_path}, model={embedding_model}")

        # 初始化 ChromaDB 客户端（持久化存储）
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 配置嵌入函数（使用 SentenceTransformer）
        self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # 创建/获取两个集合
        self.course_catalog = self._create_collection("course_catalog")  # 课程元数据
        self.course_content = self._create_collection("course_content")  # 内容片段
        logger.debug("Collections created: course_catalog, course_content")

    def _create_collection(self, name: str):
        """创建或获取 ChromaDB 集合"""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )

    def search(self,
               query: str,
               course_name: Optional[str] = None,
               lesson_number: Optional[int] = None,
               limit: Optional[int] = None) -> SearchResults:
        """
        统一搜索接口

        支持语义搜索和元数据过滤的组合查询。

        Args:
            query: 搜索查询文本
            course_name: 可选课程名称过滤（支持部分匹配）
            lesson_number: 可选课时编号过滤
            limit: 返回结果数量限制

        Returns:
            SearchResults 对象

        搜索流程：
        1. 如果提供了课程名称，先在 course_catalog 中语义匹配找到完整标题
        2. 构建元数据过滤条件
        3. 在 course_content 中执行带过滤的语义搜索
        """
        logger.debug(f"Search: query='{query[:50]}...', course='{course_name}', lesson={lesson_number}")

        # Step 1: 课程名称匹配
        course_title = None
        if course_name:
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                logger.warning(f"Course not found: '{course_name}'")
                return SearchResults.empty(f"No course found matching '{course_name}'")

        # Step 2: 构建过滤条件
        filter_dict = self._build_filter(course_title, lesson_number)
        logger.debug(f"Search filter: {filter_dict}")

        # Step 3: 执行内容搜索
        search_limit = limit if limit is not None else self.max_results

        try:
            results = self.course_content.query(
                query_texts=[query],
                n_results=search_limit,
                where=filter_dict
            )
            search_results = SearchResults.from_chroma(results)
            logger.debug(f"Search returned {len(search_results.documents)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return SearchResults.empty(f"Search error: {str(e)}")

    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """
        课程名称语义匹配

        用户可能输入部分课程名称，通过向量搜索找到完整标题。
        例如：输入 "MCP" 可匹配到 "MCP Course Introduction"

        Args:
            course_name: 用户输入的课程名称（可能不完整）

        Returns:
            匹配到的完整课程标题，或 None
        """
        try:
            results = self.course_catalog.query(
                query_texts=[course_name],
                n_results=1
            )

            if results['documents'][0] and results['metadatas'][0]:
                resolved_title = results['metadatas'][0][0]['title']
                logger.debug(f"Resolved: '{course_name}' -> '{resolved_title}'")
                return resolved_title
        except Exception as e:
            logger.error(f"Error resolving course name: {e}")

        return None

    def _build_filter(self, course_title: Optional[str], lesson_number: Optional[int]) -> Optional[Dict]:
        """
        构建 ChromaDB 过滤条件

        支持单条件和组合条件（AND 逻辑）。

        Args:
            course_title: 课程标题过滤
            lesson_number: 课时编号过滤

        Returns:
            ChromaDB where 过滤字典，或 None（无过滤）
        """
        if not course_title and lesson_number is None:
            return None

        # 同时过滤课程和课时
        if course_title and lesson_number is not None:
            return {"$and": [
                {"course_title": course_title},
                {"lesson_number": lesson_number}
            ]}

        # 仅过滤课程
        if course_title:
            return {"course_title": course_title}

        # 仅过滤课时
        return {"lesson_number": lesson_number}

    def add_course_metadata(self, course: Course):
        """
        添加课程元数据

        将课程信息存入 course_catalog 集合，用于课程名称匹配。

        Args:
            course: Course 对象
        """
        logger.debug(f"Adding course metadata: '{course.title}' ({len(course.lessons)} lessons)")

        # 序列化课时信息为 JSON
        lessons_metadata = []
        for lesson in course.lessons:
            lessons_metadata.append({
                "lesson_number": lesson.lesson_number,
                "lesson_title": lesson.title,
                "lesson_link": lesson.lesson_link
            })

        self.course_catalog.add(
            documents=[course.title],  # 用于语义匹配
            metadatas=[{
                "title": course.title,
                "instructor": course.instructor,
                "course_link": course.course_link,
                "lessons_json": json.dumps(lessons_metadata),
                "lesson_count": len(course.lessons)
            }],
            ids=[course.title]  # 使用标题作为唯一 ID
        )

    def add_course_content(self, chunks: List[CourseChunk]):
        """
        添加课程内容片段

        将分块后的课程内容存入 course_content 集合。

        Args:
            chunks: CourseChunk 列表
        """
        if not chunks:
            return

        logger.debug(f"Adding {len(chunks)} content chunks")

        documents = [chunk.content for chunk in chunks]
        metadatas = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in chunks]

        # 生成唯一 ID：课程标题_块索引
        ids = [f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}" for chunk in chunks]

        self.course_content.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def clear_all_data(self):
        """清空所有数据（删除并重建集合）"""
        try:
            logger.info("Clearing all data")
            self.client.delete_collection("course_catalog")
            self.client.delete_collection("course_content")
            self.course_catalog = self._create_collection("course_catalog")
            self.course_content = self._create_collection("course_content")
            logger.info("All data cleared")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")

    def get_existing_course_titles(self) -> List[str]:
        """获取所有已存储的课程标题"""
        try:
            results = self.course_catalog.get()
            if results and 'ids' in results:
                titles = results['ids']
                logger.debug(f"Found {len(titles)} existing courses")
                return titles
            return []
        except Exception as e:
            logger.error(f"Error getting course titles: {e}")
            return []

    def get_course_count(self) -> int:
        """获取课程总数"""
        try:
            results = self.course_catalog.get()
            count = len(results['ids']) if results and 'ids' in results else 0
            logger.debug(f"Course count: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting course count: {e}")
            return 0

    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        """
        获取所有课程的完整元数据

        解析 JSON 序列化的课时信息，返回结构化数据。
        """
        try:
            results = self.course_catalog.get()
            if results and 'metadatas' in results:
                parsed_metadata = []
                for metadata in results['metadatas']:
                    course_meta = metadata.copy()
                    if 'lessons_json' in course_meta:
                        course_meta['lessons'] = json.loads(course_meta['lessons_json'])
                        del course_meta['lessons_json']
                    parsed_metadata.append(course_meta)
                logger.debug(f"Retrieved metadata for {len(parsed_metadata)} courses")
                return parsed_metadata
            return []
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return []

    def get_course_link(self, course_title: str) -> Optional[str]:
        """获取课程链接"""
        try:
            results = self.course_catalog.get(ids=[course_title])
            if results and 'metadatas' in results and results['metadatas']:
                return results['metadatas'][0].get('course_link')
            return None
        except Exception as e:
            logger.error(f"Error getting course link: {e}")
            return None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """获取课时链接"""
        try:
            results = self.course_catalog.get(ids=[course_title])
            if results and 'metadatas' in results and results['metadatas']:
                lessons_json = results['metadatas'][0].get('lessons_json')
                if lessons_json:
                    lessons = json.loads(lessons_json)
                    for lesson in lessons:
                        if lesson.get('lesson_number') == lesson_number:
                            return lesson.get('lesson_link')
            return None
        except Exception as e:
            logger.error(f"Error getting lesson link: {e}")
            return None