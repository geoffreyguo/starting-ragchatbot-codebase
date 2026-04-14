"""
数据模型模块

定义系统中使用的核心数据结构，使用 Pydantic 进行数据验证和序列化。

核心模型：
- Lesson: 课程中的单个课时，包含编号、标题和链接
- Course: 完整课程，包含标题、链接、讲师和课时列表
- CourseChunk: 课程内容的文本片段，用于向量存储
"""

from typing import List, Optional
from pydantic import BaseModel


class Lesson(BaseModel):
    """
    课时模型

    表示课程中的一个课时/章节。

    属性：
        lesson_number: 课时编号（如 1, 2, 3 等）
        title: 课时标题
        lesson_link: 课时链接 URL（可选）
    """
    lesson_number: int
    title: str
    lesson_link: Optional[str] = None


class Course(BaseModel):
    """
    课程模型

    表示一门完整课程，包含元数据和课时列表。
    课程标题作为唯一标识符。

    属性：
        title: 课程完整标题（唯一标识）
        course_link: 课程链接 URL（可选）
        instructor: 课程讲师姓名（可选）
        lessons: 课时列表，默认为空
    """
    title: str
    course_link: Optional[str] = None
    instructor: Optional[str] = None
    lessons: List[Lesson] = []


class CourseChunk(BaseModel):
    """
    课程内容片段模型

    课程文本经分块处理后产生的片段，
    用于存入 ChromaDB 向量数据库进行语义搜索。

    属性：
        content: 片段的实际文本内容
        course_title: 所属课程标题
        lesson_number: 所属课时编号（可选）
        chunk_index: 片段在文档中的位置索引
    """
    content: str
    course_title: str
    lesson_number: Optional[int] = None
    chunk_index: int