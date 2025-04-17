# AI Assistant Guidance for Code Analysis MCP Tools

This document provides guidance on when and why to use the available tools from the `code-analysis-mcp` server for understanding and interacting with this codebase.

**Important: Codebase-Centric Workflow and the ACTIVE_CODEBASE_NAME**

This system uses a codebase-centric workflow. You manage codebases using user-defined `codebase_name`s.

* **Codebase Registry:** A central `CodebaseRegistry` collection in Weaviate stores metadata about each codebase (name, directory, status, summary, watcher status, dependencies).
* **Multi-Tenancy:** The `CodeFile` and `CodeElement` collections use Weaviate's multi-tenancy, where the `tenant_id` is the `codebase_name`.
* **Active Codebase:** The MCP server maintains an `ACTIVE_CODEBASE_NAME` variable. Most query and analysis tools operate on this active codebase context. You **must** select a codebase using `select_codebase` before using these tools.

It is crucial to be aware of the active codebase when using the tools to ensure you are interacting with the correct codebase data.

**Available Tools:**

* `scan_codebase`:
  * **When:** To add a **new** codebase to the analysis system.
  * **Why:** Creates a new codebase entry in the registry, creates the corresponding Weaviate tenant, performs the initial structural scan and upload, triggers background LLM summary generation (and enrichment if enabled), **automatically starts a file watcher**, and sets the newly scanned codebase as the active one.
  * **Args:**
    * `codebase_name` (string, required): User-defined name for the codebase. Must be unique.
    * `directory` (string, required): Absolute path of the directory containing the source code to scan.
  * **Output:** Status message indicating success/failure and if background tasks/watcher were started.
  * **Important:** This tool **errors if the `codebase_name` already exists** in the registry or if the corresponding tenant already exists in Weaviate. Use `delete_codebase` first if you need to rescan from scratch.

* `list_codebases`:
  * **When:** To see all codebases currently registered in the system.
  * **Why:** Provides an overview of available codebases, their directories, **status** (Scanning, Summarizing, Ready, Error), dependencies, and a truncated summary.
  * **Args:** None.
  * **Output:** List of codebase dictionaries.

* `select_codebase`:
  * **When:** To choose which registered codebase subsequent analysis tools should operate on. **Required before using most other tools.**
  * **Why:** Sets the `ACTIVE_CODEBASE_NAME` context within the MCP server. **Stops the watcher for the previously active codebase**, if any.
  * **Args:**
    * `codebase_name` (string, required): Name of the previously scanned codebase to set as the active context.
  * **Output:** Confirmation message including the codebase's summary.

* `delete_codebase`:
  * **When:** To completely remove a codebase and all its associated analysis data.
  * **Why:** **Stops any active file watcher** for the codebase, deletes the Weaviate tenant (clearing `CodeFile` and `CodeElement` data), and removes the entry from the `CodebaseRegistry`. Use with caution.
  * **Args:**
    * `codebase_name` (string, required): Name of the codebase whose data (including Weaviate tenant) should be deleted.
  * **Output:** Confirmation message.

* `find_element`:
  * **When:** You need to find specific code elements (functions, classes, etc.) within the **active codebase** based on name, type, or file path.
  * **Why:** Direct lookup for definitions within the selected codebase context. Available immediately after `scan_codebase`.
  * **Args:**
    * `name` (string | None, optional): Name of the code element (e.g., function name, class name) to search for.
    * `file_path` (string | None, optional): File path where the element is defined (relative to codebase root, e.g., 'src/my_module.py').
    * `element_type` (string | None, optional): Type of the element to search for (e.g., 'function', 'class', 'import').
    * `limit` (integer, optional, default 5): Maximum number of matching elements to return.
  * **Output:** Returns a concise list of matching elements: `[{ "name": ..., "type": ..., "file": ..., "uuid": ..., "description": ...}]`. The `description` field prioritizes the LLM description, falling back to the docstring.

* `get_details`:
  * **When:** You have the UUID of a specific code element (found via `find_element` or `analyze_snippet`) within the **active codebase** and need its full details.
  * **Why:** Provides comprehensive data about one specific element. Available immediately after `scan_codebase`, but LLM fields populate later if background processing is running.
  * **Args:**
    * `uuid` (string, required): The unique identifier (UUID) of the specific code element to retrieve details for.
  * **Output:** Detailed dictionary including `code_snippet`, `signature`, `parameters`, `llm_description`, `docstring`, etc.

* `analyze_snippet`:
  * **When:** You have a code snippet and want to find potentially related elements within the **active codebase**.
  * **Why:** Extracts identifiers from the snippet and uses `find_element` to locate their definitions or assignments within the active codebase. Good for understanding context around a piece of code. Available immediately after `scan_codebase`.
  * **Args:**
    * `code_snippet` (string, required): A snippet of Python code to analyze for finding related elements within the active codebase.
  * **Output:** Returns a concise list of potentially related unique elements (same format as `find_element`).

* `ask_question`:
  * **When:** You have a **specific question** about the **active codebase** (e.g., "How are user sessions managed?", "What does the `process_data` function do?"). Requires LLM features enabled and background processing to have run for relevant elements.
  * **Why:** Uses RAG - finds relevant code context via semantic search within the active codebase's tenant and uses an LLM to synthesize an answer based *only* on that context.
  * **Args:**
    * `query` (string, required): Natural language question about the codebase of the currently active codebase.
  * **Output:** Dictionary containing the LLM-generated answer: `{"answer": ...}`.

* `trigger_llm_processing`:
  * **When:** To manually start or restart background LLM enrichment/refinement for the **active codebase**. Useful after `scan_codebase` if LLMs are enabled, or if you want to force reprocessing. Requires `GENERATE_LLM_DESCRIPTIONS=true`.
  * **Why:** Provides control over the potentially long-running LLM processing for the selected codebase.
  * **Args:**
    * `uuids` (list[string] | None, optional): A specific list of element UUIDs to queue for background LLM description generation/refinement.
    * `rerun_all` (boolean, optional, default false): If true, queue all elements in the active codebase for LLM processing.
    * `skip_enriched` (boolean, optional, default true): If true, skip processing for elements that already have an LLM-generated description.
  * **Output:** Status message indicating how many elements were scheduled for background processing.

* `start_watcher`:
  * **When:** You want to manually start the automatic file watching for a specific codebase's directory (e.g., if it wasn't started automatically or was stopped).
  * **Why:** Keeps the structural analysis (AST parsing, element extraction) up-to-date for the watched codebase without needing manual rescans. Updates the `watcher_active` flag in the registry. Does *not* trigger LLM processing.
  * **Args:**
    * `codebase_name` (string, required): Name of the codebase for which to start or stop the file watcher.
  * **Output:** Confirmation message.

* `stop_watcher`:
  * **When:** You want to manually stop the automatic file watching for a specific codebase.
  * **Why:** Stops the background watcher thread for that codebase and updates the `watcher_active` flag in the registry.
  * **Args:**
    * `codebase_name` (string, required): Name of the codebase for which to start or stop the file watcher.
  * **Output:** Confirmation message.

* `add_codebase_dependency`:
  * **When:** To declare that one codebase depends on another.
  * **Why:** Enables future cross-codebase querying features.
  * **Args:**
    * `codebase_name` (string, required): The name of the codebase that has the dependency.
    * `dependency_name` (string, required): The name of the codebase it depends on.
  * **Output:** Confirmation message.

* `remove_codebase_dependency`:
  * **When:** To remove a previously declared dependency relationship.
  * **Why:** Corrects the dependency graph for future cross-codebase querying.
  * **Args:**
    * `codebase_name` (string, required): The name of the codebase to remove the dependency from.
    * `dependency_name` (string, required): The name of the dependency codebase to remove.
  * **Output:** Confirmation message.

**Workflow:**

1. Use `list_codebases` to see existing codebases or `scan_codebase` to add a new one (this also starts the watcher).
2. Use `select_codebase` to set the active codebase context (this stops the watcher for the previous codebase).
3. Use `add_codebase_dependency` or `remove_codebase_dependency` to manage relationships between codebases.
4. Use structural tools (`find_element`, `get_details`, `analyze_snippet`) to explore the active codebase's code structure. (Future: Add `include_dependencies` option).
5. If LLM features are enabled, use `ask_question` for natural language queries about the active codebase. Use `trigger_llm_processing` if you need to explicitly manage LLM enrichment. (Future: Add `include_dependencies` option).
6. Use `start_watcher` / `stop_watcher` only if you need manual control over the file watcher for a specific codebase.
7. Use `delete_codebase` to remove a codebase entirely (this also stops its watcher).
