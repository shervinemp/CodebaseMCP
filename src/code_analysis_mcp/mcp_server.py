import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from .codebase_manager import CodebaseManager

# --- Setup ---
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Global Application State ---
codebase_manager: Optional[CodebaseManager] = None

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastMCP):
    """Handles application startup and shutdown events."""
    global codebase_manager
    logger.info("--- Lifespan Start ---")

    codebase_manager = CodebaseManager()
    await codebase_manager.initialize()

    yield

    logger.info("--- Lifespan Shutdown ---")
    if codebase_manager:
        codebase_manager.shutdown()
    logger.info("--- Lifespan End ---")

# --- MCP Server Setup ---
mcp = FastMCP(
    name="code-analysis-mcp",
    description="MCP Server for Python Code Analysis using AST, Weaviate, and Gemini, with codebase management.",
    lifespan=lifespan,
)

# --- Tool Implementations (Delegating to CodebaseManager) ---

@mcp.tool(
    name="scan_codebase",
    description="Scans a directory, associates it with a codebase name, analyzes code, uploads results, generates a summary, and automatically starts a file watcher.",
)
async def scan_codebase(
    codebase_name: Annotated[str, Field(description="User-defined name for the codebase. Must be unique.")],
    directory: Annotated[str, Field(description="Absolute path of the directory containing the source code to scan.")],
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return await codebase_manager.scan_codebase(codebase_name, directory)

@mcp.tool(name="list_codebases", description="Lists all registered codebases and their status.")
async def list_codebases():
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return codebase_manager.list_codebases()

@mcp.tool(name="select_codebase", description="Sets the active codebase context for subsequent commands.")
async def select_codebase(
    codebase_name: Annotated[str, Field(description="Name of the previously scanned codebase to set as the active context.")],
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return codebase_manager.select_codebase(codebase_name)

@mcp.tool(name="delete_codebase", description="Deletes a codebase's analysis data and removes it from the registry.")
async def delete_codebase(
    codebase_name: Annotated[str, Field(description="Name of the codebase whose data should be deleted.")],
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return codebase_manager.delete_codebase(codebase_name)

@mcp.tool(name="start_watcher", description="Starts a file watcher for the specified codebase directory.")
async def start_watcher_tool(
    codebase_name: Annotated[str, Field(description="Name of the codebase for which to start the file watcher.")],
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    success, message = codebase_manager.start_watcher(codebase_name)
    return {"status": "success" if success else "error", "message": message}

@mcp.tool(name="stop_watcher", description="Stops the file watcher for the specified codebase.")
async def stop_watcher_tool(
    codebase_name: Annotated[str, Field(description="Name of the codebase for which to stop the file watcher.")],
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    success, message = codebase_manager.stop_watcher_sync(codebase_name)
    return {"status": "success" if success else "error", "message": message}

@mcp.tool(name="find_element", description="Finds code elements in the active codebase based on criteria.")
async def find_element(
    name: Annotated[Optional[str], Field(description="Name of the code element to search for.")] = None,
    file_path: Annotated[Optional[str], Field(description="File path where the element is defined.")] = None,
    element_type: Annotated[Optional[str], Field(description="Type of the element to search for.")] = None,
    limit: Annotated[int, Field(description="Maximum number of matching elements to return.")] = 5,
):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return await codebase_manager.find_element(name, file_path, element_type, limit)

@mcp.tool(name="get_details", description="Retrieves detailed information for a specific code element.")
async def get_details(uuid: Annotated[str, Field(description="The UUID of the code element.")]):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return await codebase_manager.get_details(uuid)

@mcp.tool(name="ask_question", description="Answers a question about the active codebase using RAG.")
async def ask_question(query: Annotated[str, Field(description="Natural language question about the codebase.")]):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return await codebase_manager.ask_question(query)

@mcp.tool(name="analyze_snippet", description="Analyzes a code snippet to find related elements in the active codebase.")
async def analyze_snippet(code_snippet: Annotated[str, Field(description="A snippet of Python code to analyze.")]):
    if not codebase_manager:
        return {"status": "error", "message": "Codebase manager not initialized."}
    return await codebase_manager.analyze_snippet(code_snippet)

# --- Run the server ---
if __name__ == "__main__":
    logger.info("Starting MCP server...")
    mcp.run()
