"""
搜索工具模块

提供 RAG 系统的工具定义和管理功能，支持 OpenAI function calling 机制。

核心组件：
- Tool: 工具抽象基类，定义统一的接口规范
- CourseSearchTool: 课程内容搜索工具，在 ChromaDB 中检索课程材料
- ToolManager: 工具管理器，负责工具注册、执行和状态管理
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from vector_store import VectorStore, SearchResults

logger = logging.getLogger(__name__)


class Tool(ABC):
    """
    工具抽象基类

    定义工具的标准接口，所有工具必须实现：
    - get_tool_definition(): 返回 OpenAI function calling 格式的工具定义
    - execute(): 执行工具逻辑并返回结果字符串

    这种设计便于扩展新工具类型（如天气查询、计算器等）。
    """

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        返回工具定义（OpenAI function calling 格式）

        格式结构：
        {
            "type": "function",
            "function": {
                "name": "工具名称",
                "description": "工具用途描述",
                "parameters": { JSON Schema 参数定义 }
            }
        }

        Returns:
            OpenAI 格式的工具定义字典
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        执行工具逻辑

        Args:
            **kwargs: 模型根据 parameters 定义传入的参数

        Returns:
            工具执行结果字符串，供模型生成最终响应时使用
        """
        pass


class CourseSearchTool(Tool):
    """
    课程内容搜索工具

    在 ChromaDB 向量数据库中执行语义搜索，支持：
    - 基于查询文本的语义相似度匹配
    - 按课程名称过滤（支持部分匹配）
    - 按课程编号过滤

    属性：
        store: VectorStore 实例，提供向量搜索能力
        last_sources: 最近搜索结果来源列表，供前端展示引用
    """

    def __init__(self, vector_store: VectorStore):
        """初始化搜索工具，绑定 VectorStore 实例"""
        self.store = vector_store
        self.last_sources = []  # 追踪搜索来源，供 UI 显示
        logger.debug("CourseSearchTool initialized")

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        定义 search_course_content 工具

        参数说明：
        - query (必填): 搜索内容，如 "Python basics"
        - course_name (可选): 课程名称过滤，支持部分匹配，如 "MCP"
        - lesson_number (可选): 课程编号过滤，如 1, 2, 3
        """
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str, course_name: Optional[str] = None,
                lesson_number: Optional[int] = None) -> str:
        """
        执行语义搜索

        Args:
            query: 搜索查询文本
            course_name: 可选课程名称过滤（部分匹配）
            lesson_number: 可选课程编号过滤

        Returns:
            格式化的搜索结果，或错误/无结果提示
        """
        logger.debug(f"Executing search: query='{query}', course='{course_name}', lesson={lesson_number}")

        # 调用 VectorStore 执行向量搜索
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )

        logger.debug(f"Search returned {len(results.documents)} results, error={results.error}")

        # 处理错误
        if results.error:
            logger.warning(f"Search error: {results.error}")
            return results.error

        # 处理无结果
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """
        格式化搜索结果为带来源标注的文本

        输出格式示例：
        [MCP Course - Lesson 1]
        课程内容片段...

        [Python Intro - Lesson 3]
        其他内容片段...
        """
        formatted = []
        sources = []

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # 构建来源标题
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # 记录来源供 UI 使用
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
            sources.append(source)

            formatted.append(f"{header}\n{doc}")

        self.last_sources = sources
        return "\n\n".join(formatted)


class ToolManager:
    """
    工具管理器

    负责工具生命周期管理：
    - register_tool(): 注册工具实例
    - get_tool_definitions(): 获取所有工具定义供 API 使用
    - execute_tool(): 根据名称执行工具
    - get_last_sources(): 获取最近搜索来源
    - reset_sources(): 清空来源记录

    使用流程：
    1. 创建 ToolManager
    2. 注册工具（如 CourseSearchTool）
    3. 将工具定义传给 OpenAI API
    4. 模型调用工具时，通过 execute_tool() 执行

    属性：
        tools: 工具字典，键为工具名称，值为工具实例
    """

    def __init__(self):
        """初始化空的工具字典"""
        self.tools = {}
        logger.debug("ToolManager initialized")

    def register_tool(self, tool: Tool):
        """
        注册工具

        从工具定义中提取名称作为字典键

        Args:
            tool: 实现 Tool 接口的工具实例

        Raises:
            ValueError: 工具定义缺少 name 字段
        """
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("function", {}).get("name")

        if not tool_name:
            raise ValueError("Tool must have a 'name' in its function definition")

        self.tools[tool_name] = tool
        logger.debug(f"Registered tool: {tool_name}")

    def get_tool_definitions(self) -> list:
        """
        获取所有工具定义

        Returns:
            OpenAI function calling 格式的工具定义列表
        """
        definitions = [tool.get_tool_definition() for tool in self.tools.values()]
        logger.debug(f"Getting {len(definitions)} tool definitions")
        return definitions

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果，或错误提示
        """
        if tool_name not in self.tools:
            logger.warning(f"Tool '{tool_name}' not found")
            return f"Tool '{tool_name}' not found"

        logger.debug(f"Executing tool: {tool_name}")
        result = self.tools[tool_name].execute(**kwargs)
        logger.debug(f"Tool '{tool_name}' returned {len(result)} chars")
        return result

    def get_last_sources(self) -> list:
        """
        获取最近搜索来源

        Returns:
            来源信息列表（如 ["MCP Course - Lesson 1", ...]）
        """
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                logger.debug(f"Retrieved sources: {tool.last_sources}")
                return tool.last_sources
        return []

    def reset_sources(self):
        """清空所有工具的来源记录"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
        logger.debug("Sources reset")