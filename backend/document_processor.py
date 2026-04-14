"""
文档处理模块

解析课程文档，提取结构化信息并生成文本片段用于向量存储。

核心功能：
- 读取课程文档（支持 UTF-8 编码）
- 解析课程元数据（标题、链接、讲师）
- 识别课时结构（Lesson N: Title 格式）
- 智能分块处理（基于句子边界，带重叠）
- 生成 Course 和 CourseChunk 对象

文档格式要求：
    Course Title: [课程标题]
    Course Link: [课程链接]
    Course Instructor: [讲师姓名]

    Lesson 0: [课时标题]
    Lesson Link: [课时链接]
    [课时内容...]

    Lesson 1: [课时标题]
    ...
"""

import os
import re
from typing import List, Tuple
import logging
from models import Course, Lesson, CourseChunk

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器

    将原始课程文档转换为结构化数据和可搜索的文本片段。

    属性：
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """初始化处理器，设置分块参数"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def read_file(self, file_path: str) -> str:
        """
        读取文档文件

        使用 UTF-8 编码读取文件内容，失败时使用容错模式。

        Args:
            file_path: 文档文件路径

        Returns:
            文件内容字符串
        """
        logger.debug(f"Reading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.debug(f"File read: {len(content)} characters")
                return content
        except UnicodeDecodeError:
            # UTF-8 解码失败时，忽略无法解码的字符
            logger.warning(f"UTF-8 decode error, using fallback")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                logger.debug(f"File read with fallback: {len(content)} characters")
                return content

    def chunk_text(self, text: str) -> List[str]:
        """
        文本分块处理

        按句子边界分割文本，生成带重叠的文本块。
        重叠设计确保上下文连贯，提高搜索质量。

        Args:
            text: 待分块的文本

        Returns:
            文本块列表

        分块策略：
        1. 按句子分割（识别句号、问号、感叹号后的换行）
        2. 组合句子直到达到 chunk_size
        3. 下一块从当前块末尾 overlap 字符处开始
        """
        logger.debug(f"Chunking text: {len(text)} characters")

        # 清理文本，标准化空白字符
        text = re.sub(r'\s+', ' ', text.strip())

        # 按句子分割（处理缩写词等特殊情况）
        # 正则表达式：识别句号/问号/感叹号后跟随大写字母的位置
        sentence_endings = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])'
        )
        sentences = sentence_endings.split(text)

        # 清理句子
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug(f"Split into {len(sentences)} sentences")

        chunks = []
        i = 0

        while i < len(sentences):
            current_chunk = []
            current_size = 0

            # 从句子 i 开始构建当前块
            for j in range(i, len(sentences)):
                sentence = sentences[j]
                space_size = 1 if current_chunk else 0
                total_addition = len(sentence) + space_size

                # 检查是否超出块大小限制
                if current_size + total_addition > self.chunk_size and current_chunk:
                    break

                current_chunk.append(sentence)
                current_size += total_addition

            # 添加文本块
            if current_chunk:
                chunks.append(' '.join(current_chunk))

                # 计算重叠区域
                if self.chunk_overlap > 0:
                    overlap_size = 0
                    overlap_sentences = 0

                    # 从当前块末尾倒推重叠句子数
                    for k in range(len(current_chunk) - 1, -1, -1):
                        sentence_len = len(current_chunk[k]) + (1 if k < len(current_chunk) - 1 else 0)
                        if overlap_size + sentence_len <= self.chunk_overlap:
                            overlap_size += sentence_len
                            overlap_sentences += 1
                        else:
                            break

                    # 设置下一个块的起始位置
                    next_start = i + len(current_chunk) - overlap_sentences
                    i = max(next_start, i + 1)  # 确保进度推进
                else:
                    i += len(current_chunk)
            else:
                i += 1

        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def process_course_document(self, file_path: str) -> Tuple[Course, List[CourseChunk]]:
        """
        处理课程文档

        解析文档结构，提取课程元数据和课时内容，
        生成分块后的课程片段。

        Args:
            file_path: 课程文档路径

        Returns:
            (Course 对象, CourseChunk 列表)

        解析流程：
        1. 读取文件内容
        2. 提取课程标题、链接、讲师（前三行）
        3. 识别课时标记（Lesson N: Title）
        4. 对每个课时内容进行分块
        5. 添加课程和课时上下文信息
        """
        logger.debug(f"Processing course document: {file_path}")
        content = self.read_file(file_path)
        filename = os.path.basename(file_path)

        lines = content.strip().split('\n')

        # 解析课程元数据
        course_title = filename  # 默认使用文件名
        course_link = None
        instructor_name = "Unknown"

        # 从第一行解析课程标题
        if len(lines) >= 1 and lines[0].strip():
            title_match = re.match(r'^Course Title:\s*(.+)$', lines[0].strip(), re.IGNORECASE)
            if title_match:
                course_title = title_match.group(1).strip()
            else:
                course_title = lines[0].strip()

        # 解析课程链接和讲师（前四行）
        for i in range(1, min(len(lines), 4)):
            line = lines[i].strip()
            if not line:
                continue

            # 匹配课程链接
            link_match = re.match(r'^Course Link:\s*(.+)$', line, re.IGNORECASE)
            if link_match:
                course_link = link_match.group(1).strip()
                continue

            # 匹配讲师姓名
            instructor_match = re.match(r'^Course Instructor:\s*(.+)$', line, re.IGNORECASE)
            if instructor_match:
                instructor_name = instructor_match.group(1).strip()
                continue

        # 创建 Course 对象
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=instructor_name if instructor_name != "Unknown" else None
        )
        logger.debug(f"Created course: '{course_title}', instructor='{instructor_name}'")

        # 处理课时内容
        course_chunks = []
        current_lesson = None
        lesson_title = None
        lesson_link = None
        lesson_content = []
        chunk_counter = 0

        # 跳过元数据行，从第四行开始处理课时
        start_index = 3
        if len(lines) > 3 and not lines[3].strip():
            start_index = 4  # 跳过空行

        i = start_index
        while i < len(lines):
            line = lines[i]

            # 匹配课时标记（如 "Lesson 1: Introduction"）
            lesson_match = re.match(r'^Lesson\s+(\d+):\s*(.+)$', line.strip(), re.IGNORECASE)

            if lesson_match:
                # 处理上一个课时（如果存在）
                if current_lesson is not None and lesson_content:
                    lesson_text = '\n'.join(lesson_content).strip()
                    if lesson_text:
                        # 创建 Lesson 对象
                        lesson = Lesson(
                            lesson_number=current_lesson,
                            title=lesson_title,
                            lesson_link=lesson_link
                        )
                        course.lessons.append(lesson)

                        # 对课时内容分块
                        chunks = self.chunk_text(lesson_text)
                        for idx, chunk in enumerate(chunks):
                            # 第一个块添加课时上下文
                            if idx == 0:
                                chunk_with_context = f"Lesson {current_lesson} content: {chunk}"
                            else:
                                chunk_with_context = chunk

                            course_chunk = CourseChunk(
                                content=chunk_with_context,
                                course_title=course.title,
                                lesson_number=current_lesson,
                                chunk_index=chunk_counter
                            )
                            course_chunks.append(course_chunk)
                            chunk_counter += 1

                # 开始新课时
                current_lesson = int(lesson_match.group(1))
                lesson_title = lesson_match.group(2).strip()
                lesson_link = None

                # 检查下一行是否是课时链接
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    link_match = re.match(r'^Lesson Link:\s*(.+)$', next_line, re.IGNORECASE)
                    if link_match:
                        lesson_link = link_match.group(1).strip()
                        i += 1  # 跳过链接行

                lesson_content = []
            else:
                # 添加内容到当前课时
                lesson_content.append(line)

            i += 1

        # 处理最后一个课时
        if current_lesson is not None and lesson_content:
            lesson_text = '\n'.join(lesson_content).strip()
            if lesson_text:
                lesson = Lesson(
                    lesson_number=current_lesson,
                    title=lesson_title,
                    lesson_link=lesson_link
                )
                course.lessons.append(lesson)

                chunks = self.chunk_text(lesson_text)
                for idx, chunk in enumerate(chunks):
                    # 所有块添加课程和课时上下文
                    chunk_with_context = f"Course {course_title} Lesson {current_lesson} content: {chunk}"

                    course_chunk = CourseChunk(
                        content=chunk_with_context,
                        course_title=course.title,
                        lesson_number=current_lesson,
                        chunk_index=chunk_counter
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1

        # 如果没有课时结构，将全部内容作为单个文档处理
        if not course_chunks and len(lines) > 2:
            remaining_content = '\n'.join(lines[start_index:]).strip()
            if remaining_content:
                chunks = self.chunk_text(remaining_content)
                for chunk in chunks:
                    course_chunk = CourseChunk(
                        content=chunk,
                        course_title=course.title,
                        chunk_index=chunk_counter
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1

        logger.debug(f"Processed '{course.title}': {len(course.lessons)} lessons, {len(course_chunks)} chunks")
        return course, course_chunks