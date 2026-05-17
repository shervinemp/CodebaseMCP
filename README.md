# CodebaseMCP

MCP server that scans Python codebases with AST, stores them in Weaviate, and answers questions about them via pluggable LLM providers (Gemini, OpenAI).

## Quickstart

```bash
# 1. Start Weaviate
docker-compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Create .env (gitignored) with your API key
echo "LLM_API_KEY=your_key_here" >> .env
echo "GENERATE_LLM_DESCRIPTIONS=true" >> .env

# 4. Run server
python src/code_analysis_mcp/mcp_server.py
```

Then connect via any MCP client and call `scan_codebase` with a project path.

## Features

- **AST scanning** — extracts functions, classes, imports, calls, variables, decorators, signatures, docstrings, and cross-references from Python files.
- **Vector search** — stores everything in Weaviate with multi-tenant isolation. Search by name, type, file, or semantic similarity.
- **LLM enrichment** — generates descriptions and embeddings for every function/class. Refines them using caller/callee/sibling context.
- **RAG Q&A** — `ask_question` answers natural language questions by retrieving relevant code + synthesising with an LLM.
- **File watching** — automatically rescans and re-enriches on file changes.
- **Codebase dependencies** — declare relationships between codebases; queries can span dependencies.
- **Call graph visualization** — generates MermaidJS `graph TD` from stored cross-references.
- **MCP tools** — all features exposed as MCP tools with pydantic-validated arguments.

## Configuration

Create a `.env` file in the project root (it is gitignored):

```dotenv
# --- LLM Provider ---
LLM_PROVIDER=gemini              # "gemini" (default) or "openai"
LLM_API_KEY=sk-...               # generic — works for any provider

# Or use provider-specific keys (alternative to LLM_API_KEY):
# GEMINI_API_KEY=...
# OPENAI_API_KEY=...

# --- LLM Features ---
GENERATE_LLM_DESCRIPTIONS=true   # enables enrichment, refinement, RAG
# LLM_CONCURRENCY=5               # max concurrent LLM tasks

# --- Model selection (optional) ---
# Gemini: GENERATION_MODEL_NAME="models/gemini-2.0-flash-001"
# OpenAI:  GENERATION_MODEL_NAME="gpt-4o"
# Gemini:  EMBEDDING_MODEL_NAME="models/embedding-001"
# OpenAI:  EMBEDDING_MODEL_NAME="text-embedding-3-small"

# --- Weaviate (optional) ---
# WEAVIATE_HOST=localhost
# WEAVIATE_PORT=8080
# WEAVIATE_GRPC_PORT=50051

# --- Advanced ---
# WEAVIATE_BATCH_SIZE=100
# SEMANTIC_SEARCH_LIMIT=5
# SEMANTIC_SEARCH_DISTANCE=0.7
# WATCHER_POLLING_INTERVAL=5
```

## Architecture

```
src/code_analysis_mcp/
├── llm/                  # Pluggable LLM providers
│   ├── base.py           #   LLMProvider ABC (implement to add a provider)
│   ├── gemini.py         #   GeminiProvider
│   ├── openai.py         #   OpenAIProvider
│   └── factory.py        #   Provider factory + singleton
├── code_scanner.py       # AST parsing, element extraction, enrichment
├── weaviate_client.py    # Schema, CRUD, semantic search
├── rag.py                # RAG Q&A, description refinement, summaries
├── mcp_server.py         # FastMCP server, tools, watcher, lifespan
├── visualization.py      # MermaidJS call graphs
└── utils.py              # Shared helpers
```

### LLM provider system

Add a new provider by:
1. Creating a class in `llm/` that implements `LLMProvider` (generate + embed + name + is_available)
2. Adding it to the factory in `llm/factory.py`

The provider is selected at startup via `LLM_PROVIDER`. The API key is read from `LLM_API_KEY` (generic) or the provider-specific fallback (`GEMINI_API_KEY`, `OPENAI_API_KEY`).

### Data model

Weaviate uses three collections:
- **CodeFile** (multi-tenant) — file paths and modification times.
- **CodeElement** (multi-tenant) — every parsed function, class, import, call, and variable assignment, with optional LLM-generated vector embeddings.
- **CodebaseRegistry** (global) — tracks codebase names, directories, scan status, summaries, watcher status, and dependency lists.

Each codebase is isolated in its own tenant (tenant ID = codebase name). Cross-codebase queries look up declared dependencies in the registry and fan out across tenants.

## MCP Tools

| Tool | Description |
|------|-------------|
| `scan_codebase` | Scan a directory, upload to Weaviate, generate summary, start watcher |
| `list_codebases` | List all registered codebases and their status |
| `select_codebase` | Set the active codebase for subsequent queries |
| `delete_codebase` | Remove a codebase from Weaviate and the registry |
| `find_element` | Search elements by name, file, type across active + dependencies |
| `get_details` | Full properties for a specific element UUID |
| `analyze_snippet` | Find elements related to a code snippet via identifier matching |
| `ask_question` | RAG question answering against the active codebase |
| `trigger_llm_processing` | Manually queue elements for LLM enrichment |
| `regenerate_summary` | Re-run summary generation for the active codebase |
| `start_watcher` / `stop_watcher` | Control file watching manually |
| `add_codebase_dependency` / `remove_codebase_dependency` | Manage dependency graph |
