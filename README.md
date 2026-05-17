# CodebaseMCP

MCP server that scans Python codebases with AST, stores them in Weaviate, and answers questions via pluggable LLM providers (Gemini, OpenAI).

## Quickstart

```bash
docker-compose up -d
pip install -r requirements.txt
echo "LLM_API_KEY=your_key" >> .env
echo "GENERATE_LLM_DESCRIPTIONS=true" >> .env
python src/code_analysis_mcp/mcp_server.py
```

Call `scan_codebase` from any MCP client with a project path.

## Features

- **AST scanning** — functions, classes, imports, calls, variables, decorators, signatures, docstrings, cross-references.
- **Hybrid search** — Weaviate vector + BM25 keyword search. Unenriched elements are still findable by name/code content.
- **DAG-ordered enrichment** — callees enriched before callers, so refinement sees real callee descriptions.
- **RAG Q&A** — `ask_question` retrieves relevant code via hybrid search and synthesises answers with an LLM.
- **File watching** — rescans and re-enriches on file changes, DAG-ordered.
- **Worker pool** — enrichment uses `LLM_CONCURRENCY` workers with a queue instead of N asyncio tasks.
- **Codebase dependencies** — relationships between codebases; queries span dependencies in parallel.
- **Call graph visualization** — MermaidJS from stored cross-references.

## Configuration

```dotenv
LLM_PROVIDER=gemini                     # or "openai"
LLM_API_KEY=                            # generic key for any provider
GENERATE_LLM_DESCRIPTIONS=true          # enables enrichment, refinement, RAG

LLM_CONCURRENCY=5                       # worker pool size
GENERATION_MODEL_NAME=models/gemini-3.1-flash-lite-preview
EMBEDDING_MODEL_NAME=models/gemini-embedding-2
WEAVIATE_HOST=localhost                 # these 3 env vars are now functional
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
SEMANTIC_SEARCH_LIMIT=10
SEMANTIC_SEARCH_DISTANCE=0.7
WATCHER_POLLING_INTERVAL=5
```

## Architecture

```
src/code_analysis_mcp/
├── llm/                  # Pluggable LLM providers
│   ├── base.py           #   ABC: implement generate() + embed()
│   ├── gemini.py         #   GeminiProvider
│   ├── openai.py         #   OpenAIProvider
│   └── factory.py        #   Factory + singleton, reads LLM_PROVIDER
├── code_scanner.py       # AST parsing, element extraction, upload
├── weaviate_client.py    # Schema, CRUD, hybrid search
├── rag.py                # RAG Q&A, description refinement, summaries
├── tasks.py              # Enrichment workers, DAG sort, summary (extracted from mcp_server)
├── mcp_server.py         # FastMCP server, tools, watcher, lifespan
├── visualization.py      # MermaidJS call graphs
└── utils.py              # Shared helpers
```

### LLM provider system

New providers implement the `LLMProvider` ABC in `llm/base.py` and register in `llm/factory.py`. Key selected via `LLM_API_KEY` (generic) or `GEMINI_API_KEY` / `OPENAI_API_KEY` (fallback).

### Data model

- **CodeFile** (multi-tenant) — file paths, modification times.
- **CodeElement** (multi-tenant) — every parsed element with optional vector.
- **CodebaseRegistry** (global) — codebase metadata, status, dependencies.

Tenant ID = codebase name. Cross-codebase queries fan out across dependency tenants.

## MCP Tools

| Tool | Description |
|------|-------------|
| `scan_codebase` | Scan, upload, DAG-enrich, summarise, start watcher |
| `list_codebases` | Registered codebases with status |
| `select_codebase` | Set active context (stops prior watcher) |
| `delete_codebase` | Remove codebase + tenant + registry entry |
| `find_element` | Search by name, file, type across active + dependencies |
| `get_details` | Full properties for a UUID |
| `analyze_snippet` | Find elements related to a code snippet |
| `ask_question` | RAG (with optional `include_dependencies`) |
| `trigger_llm_processing` | Queue enrichment (DAG-ordered, worker pool) |
| `regenerate_summary` | Re-run summary without rescanning |
| `start_watcher` / `stop_watcher` | Manual watcher control |
| `add_codebase_dependency` / `remove_codebase_dependency` | Dependency graph management |
