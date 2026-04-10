from openai import OpenAI
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with OpenAI-compatible API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
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
        # Support custom base_url for third-party endpoints
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build messages list
        messages = []

        # Add system message
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages.append({"role": "system", "content": system_content})

        # Add user query
        messages.append({"role": "user", "content": query})

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }

        # Convert Anthropic tool format to OpenAI format if needed
        if tools:
            openai_tools = self._convert_tools_to_openai(tools)
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        # Get response
        response = self.client.chat.completions.create(**api_params)

        # Handle tool execution if needed
        message = response.choices[0].message
        if message.tool_calls and tool_manager:
            return self._handle_tool_execution(message, messages, openai_tools, tool_manager)

        # Return direct response
        return message.content or ""

    def _convert_tools_to_openai(self, anthropic_tools: List) -> List[Dict]:
        """Convert Anthropic tool format to OpenAI format"""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        return openai_tools

    def _handle_tool_execution(self, message, messages: List, tools: List, tool_manager) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            message: The response message containing tool calls
            messages: Current conversation messages
            tools: Available tools
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Add assistant message with tool calls
        messages.append(message)

        # Execute all tool calls and collect results
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            # Parse arguments from JSON string
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Execute the tool
            tool_result = tool_manager.execute_tool(tool_name, **args)

            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Get final response without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }

        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content or ""