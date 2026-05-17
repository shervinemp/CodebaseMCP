# CodebaseMCP Agent Guidance

Python code analysis MCP server. Uses AST parsing, Weaviate vector store, pluggable LLM providers (Gemini/OpenAI).

## Multi-Tenancy

Each codebase gets a Weaviate tenant (ID = `codebase_name`). The server tracks an `ACTIVE_CODEBASE_NAME`. Most tools operate on this active context — call `select_codebase` first.

## Key Concepts

- **Hybrid search** — uses both vectors and BM25 keyword matching. Elements without LLM embeddings are still findable by name/code content.
- **DAG-ordered enrichment** — callees enriched before callers so refinement sees real descriptions for dependencies.
- **Worker pool** — enrichment uses `LLM_CONCURRENCY` workers consuming a queue instead of N asyncio tasks.
- **`LLM_API_KEY`** — generic key env var (falls back to `GEMINI_API_KEY` / `OPENAI_API_KEY`).

## Tools

| Tool | When to use |
|------|-------------|
| `scan_codebase` | Add a new codebase. Creates tenant, scans AST, uploads, starts watcher, DAG-ordered enrichment via worker pool. Errors if name exists. |
| `list_codebases` | See all registered codebases with status and summary. |
| `select_codebase` | Required before most tools. Sets active context (stops watcher for previous). |
| `delete_codebase` | Remove a codebase and its data entirely. |
| `find_element` | Search functions/classes by name, file, type. |
| `get_details` | Full details (snippet, signature, LLM description) for a specific element UUID. |
| `analyze_snippet` | Extract identifiers from a code snippet and find related elements. |
| `ask_question` | RAG question answering. Use `include_dependencies=False` to skip dependency codebases. |
| `trigger_llm_processing` | Manually queue elements for LLM enrichment (DAG-ordered, worker pool). |
| `regenerate_summary` | Regenerate the codebase summary without rescanning. |
| `start_watcher` / `stop_watcher` | Manual watcher control (starts automatically on scan, triggers DAG-ordered enrichment on rescans). |
| `add_codebase_dependency` / `remove_codebase_dependency` | Declare cross-codebase relationships for combined queries. |

## Workflow

1. `scan_codebase` to add a codebase (creates tenant, starts watcher, triggers DAG-ordered enrichment via worker pool).
2. `select_codebase` to set the active context.
3. `find_element` / `get_details` / `analyze_snippet` for structural queries.
4. `ask_question` for RAG (after enrichment starts — hybrid search finds unenriched elements too).
5. `trigger_llm_processing(rerun_all=True)` if you need to force re-enrichment.
