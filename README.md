# Python Codebase Analysis RAG System

This system analyzes Python code using Abstract Syntax Trees (AST), stores the extracted information (functions, classes, calls, variables, etc.) in a Weaviate vector database, and provides tools for querying and understanding the codebase via a Model Context Protocol (MCP) server. It leverages Google's Gemini models for generating embeddings and natural language descriptions/answers.

## Features

* **Code Scanning:** Parses Python files to identify code elements (functions, classes, imports, calls, assignments) and their relationships. Extracts:
  * Basic info: Name, type, file path, line numbers, code snippet, docstring.
  * Function/Method details: Parameters, return type, signature, decorators.
  * Scope info: Parent scope (class/function) UUID, readable ID (e.g., `file:type:name:line`), base class names.
  * Usage info: Attribute accesses within scopes, call relationships (partially tracked).
* **Vector Storage:** Uses Weaviate to store code elements and their vector embeddings (when LLM generation is enabled).
* **LLM Enrichment (Optional & Background):** Generates semantic descriptions and embeddings for functions and classes using Gemini. This now runs as background tasks triggered after scanning or manually. Can be enabled/disabled via the `.env` file.
* **Automatic Refinement (Optional & Background):** When LLM generation is enabled, automatically refines descriptions for new/updated functions using context (callers, callees, siblings, related variables) as part of the background processing.
* **RAG Q&A:** Answers natural language questions about the codebase using Retrieval-Augmented Generation (requires LLM features enabled and background processing completed).
* **User Clarifications:** Allows users to add manual notes to specific code elements.
* **Visualization:** Generates MermaidJS call graphs based on stored relationships.
* **MCP Server:** Exposes analysis and querying capabilities through MCP tools, managing codebases and an active codebase context.
* **File Watcher (Integrated):** Automatically starts when a codebase is scanned (`scan_codebase`) and stops when another codebase is selected (`select_codebase`) or the codebase is deleted (`delete_codebase`). Triggers re-analysis and database updates for the *active* codebase when its files change. Can also be manually controlled via `start_watcher` and `stop_watcher` tools.
* **Codebase Dependencies:** Allows defining dependencies between scanned codebases (`add_codebase_dependency`, `remove_codebase_dependency`).
* **Cross-Codebase Querying:** Enables searching (`find_element`) and asking questions (`ask_question`) across the active codebase and its declared dependencies.

## Setup

1. **Environment:** Ensure Python 3.10+ and Docker are installed.
2. **Weaviate:** Start the Weaviate instance using Docker Compose:

    ```bash
    docker-compose up -d
    ```

3. **Dependencies:** Install Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. **API Key & Configuration:** Create a `.env` file in the project root and add your Gemini API key. You can also configure other settings:

    ```dotenv
    # --- Required ---
    GEMINI_API_KEY=YOUR_API_KEY_HERE

    # --- Optional ---
    # Set to true to enable background LLM description generation and refinement
    GENERATE_LLM_DESCRIPTIONS=true
    # Max concurrent background LLM tasks (embeddings/descriptions/refinements)
    LLM_CONCURRENCY=5
    # ANALYZE_ON_STARTUP is no longer used. Scanning is done via the scan_codebase tool.

    # Specify Weaviate connection details if not using defaults
    # WEAVIATE_HOST=localhost
    # WEAVIATE_PORT=8080
    # WEAVIATE_GRPC_PORT=50051

    # Specify alternative Gemini models if desired
    # GENERATION_MODEL_NAME="models/gemini-pro"
    # EMBEDDING_MODEL_NAME="models/embedding-001"

    # Adjust Weaviate batch size
    # WEAVIATE_BATCH_SIZE=100

    # SEMANTIC_SEARCH_LIMIT=5
    # SEMANTIC_SEARCH_DISTANCE=0.7
    # Watcher polling interval (seconds)
    # WATCHER_POLLING_INTERVAL=5
    ```

5. **Run MCP Server:** Start the server in a separate terminal:

    ```bash
    python src/code_analysis_mcp/mcp_server.py
    ```

    *(Ensure this terminal stays running for the tools to be available)*

## Architecture Overview

This system analyzes Python code, stores the extracted information in a Weaviate vector database, and provides tools for querying and understanding the codebase via a Model Context Protocol (MCP) server. It leverages Google's Gemini models for generating embeddings and natural language descriptions/answers.

The main modules are:

* `code_scanner.py`: Finds Python files, parses them using AST, extracts structural elements (functions, classes, imports, calls, etc.), and prepares data for Weaviate.
* `weaviate_client.py`: Manages the connection to Weaviate, defines the data schema (`CodeFile`, `CodeElement`, `CodebaseRegistry`), and provides functions for batch uploading, querying, updating, and deleting data.
* `rag.py`: Implements Retrieval-Augmented Generation (RAG) for answering questions about the codebase. It uses semantic search to find relevant code elements and an LLM to synthesize an answer.
* `mcp_server.py`: Sets up the FastMCP server, manages codebases in a `CodebaseRegistry` collection, handles the active codebase context (`ACTIVE_CODEBASE_NAME`), integrates file watching logic (including automatic start/stop), manages codebase dependencies, and exposes analysis functionalities as MCP tools with detailed argument descriptions.
* `visualization.py`: Generates MermaidJS call graphs based on stored relationships.

The system uses Weaviate's multi-tenancy feature for `CodeFile` and `CodeElement` collections, where the tenant ID is the user-defined `codebase_name`. A separate, non-multi-tenant `CodebaseRegistry` collection tracks codebase metadata (name, directory, status, summary, watcher status, dependencies). The `ACTIVE_CODEBASE_NAME` global variable in the server determines the primary codebase tenant for queries. Query tools (`find_element`, `ask_question`) can optionally search across the active codebase and its declared dependencies stored in the registry. The `list_codebases` tool can be used to view the status and dependencies of all codebases.

Background LLM processing is used to generate semantic descriptions and embeddings for code elements. This is an optional feature that can be enabled/disabled via the `.env` file.

Detailed information on the available tools and their arguments can be retrieved directly from the MCP server using standard MCP introspection methods once the server is running.
