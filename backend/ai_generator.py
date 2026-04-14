"""
AI 生成器模块

封装 OpenAI API 调用，支持 function calling（工具调用）机制。

核心功能：
- 调用 OpenAI 兼容 API 生成响应
- 支持 tool calling：模型可自主决定调用搜索工具
- 处理工具调用流程：接收 tool_calls → 执行工具 → 继续对话
- 对话历史管理：支持多轮对话上下文

Function Calling 流程：
    1. 发送用户查询 + 工具定义给模型
    2. 模型决定调用工具，返回 tool_calls
    3. 执行工具获取结果
    4. 将工具结果发送给模型
    5. 模型生成最终回答

使用示例：
    generator = AIGenerator(api_key, model, base_url)
    response = generator.generate_response(
        query="MCP 是什么?",
        tools=[tool_definitions],
        tool_manager=tool_manager
    )
"""

from openai import OpenAI
from typing import List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class AIGenerator:
    """
    AI 生成器

    与 OpenAI 兼容 API 交互，支持工具调用。

    属性：
        client: OpenAI 客户端实例
        model: 模型名称
        base_params: 基础 API 参数（model, temperature, max_tokens）
    """

    # 系统提示词（指导模型行为）
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked."""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        """
        初始化 AI 生成器

        Args:
            api_key: API 密钥
            model: 模型名称（如 gpt-4o-mini）
            base_url: API 基础 URL（第三方兼容端点）
        """
        logger.debug(f"Initializing AIGenerator: model={model}, base_url={base_url}")

        # 配置客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model

        # 预构建基础参数
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        logger.debug(f"Base params: {self.base_params}")

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        生成 AI 响应

        支持工具调用和对话上下文。

        Args:
            query: 用户问题
            conversation_history: 格式化的对话历史
            tools: 工具定义列表（OpenAI function calling 格式）
            tool_manager: 工具管理器（执行工具调用）

        Returns:
            生成的响应文本
        """
        logger.debug(f"Generating response for: '{query[:50]}...'")

        # 构建消息列表
        messages = []

        # 添加系统消息（包含对话历史）
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages.append({"role": "system", "content": system_content})
        logger.debug(f"System message: {len(system_content)} chars")

        # 添加用户消息
        messages.append({"role": "user", "content": query})

        # 构建 API 参数
        api_params = {
            **self.base_params,
            "messages": messages
        }

        # 添加工具定义（已为 OpenAI 格式）
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
            logger.debug(f"Tools: {[t['function']['name'] for t in tools]}")

        # 调用 API
        logger.debug("Calling OpenAI API...")
        logger.debug(f"API params: model={api_params.get('model')}, "
                     f"temperature={api_params.get('temperature')}, "
                     f"max_tokens={api_params.get('max_tokens')}, "
                     f"messages={len(messages)}, tools={api_params.get('tools')}")
        logger.debug(f"Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")

        response = self.client.chat.completions.create(**api_params)
        logger.debug(f"Response received: finish_reason={response.choices[0].finish_reason}")

        # 处理工具调用
        message = response.choices[0].message
        if message.tool_calls and tool_manager:
            logger.debug(f"Tool calls detected: {len(message.tool_calls)}")
            return self._handle_tool_execution(message, messages, tool_manager)

        # 直接返回响应
        content = message.content or ""
        logger.debug(f"Direct response: '{content[:100]}...'")
        return content

    def _handle_tool_execution(self, message, messages: List, tool_manager) -> str:
        """
        处理工具调用流程

        流程：
        1. 将 assistant 消息（含 tool_calls）加入对话
        2. 执行每个工具调用
        3. 将工具结果加入对话
        4. 再次调用 API 获取最终响应

        Args:
            message: 包含 tool_calls 的响应消息
            messages: 当前对话消息列表
            tool_manager: 工具管理器

        Returns:
            最终响应文本
        """
        logger.debug(f"Handling tool execution, messages: {len(messages)}")

        # 添加 assistant 消息（转为字典格式以便序列化）
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        }
        messages.append(assistant_message)

        # 执行工具调用
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            # 解析参数
            try:
                args = json.loads(tool_call.function.arguments)
                logger.debug(f"Tool call: {tool_name} with args: {args}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse args: {tool_call.function.arguments}")
                args = {}

            # 执行工具
            tool_result = tool_manager.execute_tool(tool_name, **args)
            logger.debug(f"Tool result: '{tool_result[:1000]}...'")

            # 添加工具结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # 获取最终响应（不再传递工具，强制模型直接回答）
        final_params = {
            **self.base_params,
            "messages": messages,
            # 不传递 tools，模型必须直接回答
        }
        logger.debug("Calling API for final response...")
        logger.debug(f"Final params: model={final_params.get('model')}, "
                     f"messages={len(messages)}")
        logger.debug(f"Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")

        final_response = self.client.chat.completions.create(**final_params)

        # 详细调试：检查最终响应结构
        final_choice = final_response.choices[0]
        final_message = final_choice.message
        logger.debug(f"Final response details:")
        logger.debug(f"  - finish_reason: {final_choice.finish_reason}")
        logger.debug(f"  - message.role: {final_message.role}")
        logger.debug(f"  - message.content: {repr(final_message.content)}")
        logger.debug(f"  - message.tool_calls: {final_message.tool_calls}")

        # 如果模型仍然想调用工具（第三方模型兼容性问题）
        # 直接返回工具结果作为回答，避免无限循环
        if final_message.tool_calls:
            logger.warning(f"Model requested another tool call (ignoring): {[tc.function.name for tc in final_message.tool_calls]}")
            # 返回工具结果摘要作为回答
            tool_results_summary = []
            for msg in messages:
                if msg.get("role") == "tool":
                    tool_results_summary.append(msg.get("content", ""))
            if tool_results_summary:
                logger.info("Returning tool results as response due to model compatibility issue")
                return "\n\n".join(tool_results_summary[:2])  # 返回前两个工具结果

        content = final_message.content or ""
        logger.debug(f"Final response content length: {len(content)}")
        return content