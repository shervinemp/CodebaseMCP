# CodebaseMCP Agent Guidance

This codebase is a Python code analysis MCP server. Uses AST parsing, Weaviate vector storage, and pluggable LLM providers (Gemini/OpenAI).

## Multi-Tenancy

Each scanned codebase gets a Weaviate tenant (tenant ID = `codebase_name`). The server tracks an `ACTIVE_CODEBASE_NAME`. Most tools operate on this active context — call `select_codebase` first.

## Tools

| Tool | When to use |
|------|-------------|
| `scan_codebase` | Add a new codebase. Creates tenant, scans AST, uploads, starts watcher, triggers LLM enrichment. Errors if name exists. |
| `list_codebases` | See all registered codebases with status and summary. |
| `select_codebase` | Required before most tools. Sets active context (stops watcher for previous). |
| `delete_codebase` | Remove a codebase and its data entirely. |
| `find_element` | Search functions/classes by name, file, type. |
| `get_details` | Full details (snippet, signature, LLM description) for a specific element UUID. |
| `analyze_snippet` | Extract identifiers from a code snippet and find related elements. |
| `ask_question` | RAG question answering. Requires LLM features enabled and elements enriched. |
| `trigger_llm_processing` | Manually queue elements for LLM enrichment (`rerun_all` or specific `uuids`). |
| `regenerate_summary` | Regenerate the codebase summary without rescanning. |
| `start_watcher` / `stop_watcher` | Manual watcher control (starts automatically on scan). |
| `add_codebase_dependency` / `remove_codebase_dependency` | Declare cross-codebase relationships for combined queries. |

## Workflow

1. `scan_codebase` to add a codebase (creates tenant, starts watcher, triggers enrichment).
2. `select_codebase` to set the active context.
3. `find_element` / `get_details` / `analyze_snippet` for structural queries.
4. `ask_question` for RAG (after enrichment completes).
5. `trigger_llm_processing(rerun_all=True)` if you need to force re-enrichment.
