# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Quick start
./run.sh

# Or manually from backend directory
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web interface: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Dependencies

Uses `uv` as the package manager (not pip). Install dependencies:
```bash
uv sync
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials. The system uses Claude with tool calling to search ChromaDB and generate responses.

### Core Flow
```
User Query → FastAPI → RAGSystem → Claude API
                                  ↓ (tool_use)
                        CourseSearchTool → VectorStore.search()
                                  ↓
                        ChromaDB (semantic search)
                                  ↓
                        Claude synthesizes response
```

### Key Components (backend/)

- **`app.py`** - FastAPI entry point, serves frontend and API endpoints (`/api/query`, `/api/courses`)
- **`rag_system.py`** - Orchestrator coordinating all components
- **`ai_generator.py`** - Claude API integration with tool calling support
- **`vector_store.py`** - ChromaDB collections: `course_catalog` (metadata) and `course_content` (chunks)
- **`search_tools.py`** - `CourseSearchTool` registered with `ToolManager`, executed when Claude calls `search_course_content`
- **`document_processor.py`** - Parses course documents into chunks with lesson metadata
- **`session_manager.py`** - Conversation history for contextual responses

### Tool Calling Pattern

Claude is given a `search_course_content` tool with parameters: `query`, `course_name` (optional), `lesson_number` (optional). When Claude decides to search, `AIGenerator._handle_tool_execution()` runs the tool and feeds results back to Claude for final response synthesis.

## Configuration

Settings in `backend/config.py` (dataclass):
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation messages
- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2`

Environment: Requires `ANTHROPIC_API_KEY` in `.env` file.

## Document Format

Course documents in `docs/` should follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
...
```

Documents are chunked by sentence boundaries with overlap, stored with metadata (course_title, lesson_number, chunk_index).