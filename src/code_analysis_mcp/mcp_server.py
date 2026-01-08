import asyncio
import os
import contextlib
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Annotated, Optional, List

from weaviate.classes.tenants import Tenant
import re
import ast
from weaviate.exceptions import UnexpectedStatusCodeError

# --- Watcher Imports ---
import time
import threading
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileModifiedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
)

from code_scanner import (
    enrich_element,
    enrich_elements_batch,
    _scan_cleanup_and_upload,
)
from file_watcher import (
    start_watcher,
    stop_watcher,
)
from weaviate_client import (
    create_weaviate_client,
    create_schema,
    find_element_by_name,
    get_element_details,
    delete_elements_by_file_path,
    delete_code_file,
    add_codebase_registry_entry,
    update_codebase_registry,
    get_codebase_details,
    get_all_codebases,
    delete_codebase_registry_entry,
    delete_tenant,
)
from rag import (
    answer_codebase_question,
    refine_element_description,
    generate_codebase_summary,
)

# Setup logging
print("Attempting logging.basicConfig...")
logger = logging.getLogger(__name__)
log_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "mcp_server.log")
log_file_path = os.path.abspath(log_file_path)
print(f"Attempting to log to: {log_file_path}")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file_path,
    filemode="a",
)
print("logging.basicConfig finished.")
logger.critical("--- MCP SERVER LOGGING INITIALIZED ---")

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded for MCP server.")

# --- Configuration ---
LLM_ENABLED = os.getenv("GENERATE_LLM_DESCRIPTIONS", "false").lower() == "true"
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))
WATCHER_POLLING_INTERVAL = int(os.getenv("WATCHER_POLLING_INTERVAL", "5"))

if LLM_ENABLED and not os.getenv("GEMINI_API_KEY"):
    logger.warning(
        "GENERATE_LLM_DESCRIPTIONS is true in .env, but GEMINI_API_KEY is missing. LLM features will be disabled."
    )
    LLM_ENABLED = False


# --- Global State ---
global_weaviate_client = None
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
background_llm_tasks = set()
ACTIVE_CODEBASE_NAME: str | None = None
ACTIVE_WATCHERS: Dict[str, Dict[str, Any]] = {}


# --- Background Task Functions ---
# process_element_llm removed - superseded by _process_llm_batch


async def background_generate_summary(client, codebase_name: str):
    """Background task to generate and update codebase summary."""
    logger.info(
        f"Background task started: Generating summary for codebase '{codebase_name}'"
    )
    summary = await generate_codebase_summary(client, codebase_name)
    if not summary.startswith("Error:"):
        update_success = update_codebase_registry(
            client, codebase_name, {"summary": summary, "status": "Ready"}
        )
        if update_success:
            logger.info(
                f"Successfully updated summary for codebase '{codebase_name}'. Status set to Ready."
            )
        else:
            logger.error(
                f"Failed to update summary in registry for codebase '{codebase_name}'."
            )
    else:
        logger.error(
            f"Failed to generate summary for codebase '{codebase_name}': {summary}"
        )
        update_codebase_registry(client, codebase_name, {"status": "Error"})
        logger.info(
            f"Set status to Error for codebase '{codebase_name}' due to summary failure."
        )


# --- File Watcher Logic ---
# Logic moved to file_watcher.py
# Use start_watcher and stop_watcher imported from file_watcher.py
# We need to bridge the "manager" interface expected by file_watcher
# to the global_weaviate_client here.

class WeaviateManagerBridge:
    """Bridges calls from FileWatcher to the Weaviate client functions."""
    def __init__(self, client):
        self.client = client
        # Simple cache for tenant checks to avoid roundtrips (Scenario 24)
        self._tenant_cache = set()

    def get_codebase_details(self, codebase_name):
        return get_codebase_details(self.client, codebase_name)

    def update_codebase_registry(self, codebase_name, updates):
        return update_codebase_registry(self.client, codebase_name, updates)

    def delete_elements_by_file_path(self, codebase_name, path):
        return delete_elements_by_file_path(self.client, codebase_name, path)

    def delete_code_file(self, codebase_name, path):
        return delete_code_file(self.client, codebase_name, path)

    def tenant_exists(self, tenant_id):
        if tenant_id in self._tenant_cache:
            return True
        exists = self.client.collections.get("CodeElement").tenants.exists(tenant_id)
        if exists:
            self._tenant_cache.add(tenant_id)
        return exists

    def queue_llm_processing(self, codebase_name, uuids, skip_enriched=True):
        """Adds LLM tasks to the background queue."""
        logger.info(f"Queueing LLM processing for {len(uuids)} items in {codebase_name}")
        # This needs to schedule tasks on the main event loop
        # Since this method might be called from a thread, we use run_coroutine_threadsafe if needed,
        # but here we are just adding to a set and creating tasks.
        # However, asyncio.create_task must be called inside a loop.

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("No running loop found to schedule LLM tasks.")
            return False

        # We can trigger batch processing here
        task = loop.create_task(
            _process_llm_batch(self.client, codebase_name, uuids)
        )
        background_llm_tasks.add(task)
        task.add_done_callback(background_llm_tasks.discard)
        return True

async def _process_llm_batch(client, tenant_id, uuids, refine=False):
    """Internal helper to run batch enrichment and optional refinement."""
    try:
        result = await enrich_elements_batch(client, tenant_id, uuids)

        if refine:
            # We don't have list of successful uuids easily from enrich_elements_batch
            # unless we modify it to return them.
            # Currently returns counts.
            # Assuming we proceed with refinement for all, handling errors individually.
            # However, refine_element_description checks if element exists.

            # Since refinement is slow, we might want to batch this too or just iterate.
            # refine_element_description is one-by-one.
            logger.info(f"Starting refinement for {len(uuids)} elements in {tenant_id}")
            for uuid in uuids:
                # We reuse process_element_llm's refinement logic or call direct
                # process_element_llm calls enrich then refine.
                # Here we already enriched.
                # Let's call refine_element_description directly.
                try:
                    async with llm_semaphore:
                         await refine_element_description(client, tenant_id, uuid)
                except Exception as e:
                    logger.error(f"Refinement failed for {uuid}: {e}")

    except Exception as e:
        logger.error(f"Error in background LLM batch: {e}")

# Override start/stop watcher to use the bridge
async def start_watcher_wrapper(codebase_name: str) -> tuple[bool, str]:
    if not global_weaviate_client:
         return False, "Weaviate client not initialized"
    manager = WeaviateManagerBridge(global_weaviate_client)
    # Capture the running loop to pass to the watcher thread
    loop = asyncio.get_running_loop()
    # start_watcher is synchronous but launches a thread
    return start_watcher(manager, ACTIVE_WATCHERS, codebase_name, loop=loop)

async def stop_watcher_wrapper(codebase_name: str) -> tuple[bool, str]:
    if not global_weaviate_client:
         return False, "Weaviate client not initialized"
    manager = WeaviateManagerBridge(global_weaviate_client)
    return stop_watcher(manager, ACTIVE_WATCHERS, codebase_name)


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastMCP):
    """Handles application startup and shutdown events."""
    global global_weaviate_client
    logger.info("--- Lifespan Start ---")
    logger.info("MCP server starting up...")

    if not global_weaviate_client:
        logger.info("Lifespan: Creating global Weaviate client instance...")
        global_weaviate_client = create_weaviate_client()
        logger.info("Lifespan: Global Weaviate client instance created.")
    else:
        logger.info("Lifespan: Using existing global Weaviate client instance.")

    if global_weaviate_client:
        logger.info(
            "Lifespan: Weaviate client exists. Proceeding with connection logic."
        )
        try:
            logger.info("Lifespan: Waiting briefly before connecting to Weaviate...")
            await asyncio.sleep(5)

            logger.info("Lifespan: Connecting global Weaviate client...")
            global_weaviate_client.connect()
            logger.info("Lifespan: connect() method called.")

            logger.info("Lifespan: Starting readiness check loop...")
            is_ready = False
            for attempt in range(5):
                logger.info(f"Lifespan: Readiness check attempt {attempt+1}/5...")
                try:
                    if global_weaviate_client.is_ready():
                        is_ready = True
                        logger.info("Lifespan: Weaviate client is ready.")
                        break
                    logger.warning(
                        f"Lifespan: Weaviate client not ready yet (attempt {attempt+1}/5). Waiting..."
                    )
                    await asyncio.sleep(1)
                except Exception as ready_e:
                    logger.error(
                        f"Lifespan: Error during readiness check (attempt {attempt+1}/5): {ready_e}"
                    )
                    await asyncio.sleep(1)

            logger.info(f"Lifespan: Readiness check loop finished. is_ready={is_ready}")

            if is_ready:
                logger.info("Lifespan: Global Weaviate client connected and ready.")
                logger.info("Lifespan: Ensuring Weaviate schema exists...")
                try:
                    logger.info("Lifespan: Calling asyncio.to_thread(create_schema)...")
                    await asyncio.to_thread(create_schema, global_weaviate_client)
                    logger.info("Lifespan: asyncio.to_thread(create_schema) completed.")
                    logger.info(
                        "Lifespan: Schema check/creation should be complete (check weaviate_client logs)."
                    )
                    logger.info(
                        "Lifespan: MCP Server is now connected and ready to receive requests."
                    )
                except Exception as schema_exc:
                    logger.exception(
                        f"Lifespan: EXCEPTION during schema creation: {schema_exc}"
                    )
                    logger.error(
                        "Lifespan: MCP Server startup failed due to schema error."
                    )
            else:
                logger.error(
                    "Lifespan: Failed to connect or Weaviate did not become ready."
                )
        except Exception as e:
            logger.exception(
                f"Lifespan: Error during Weaviate client connection/startup analysis: {e}"
            )
    else:
        logger.error(
            "Lifespan: Failed to create global Weaviate client instance during startup."
        )

    logger.info(
        f"--- Lifespan Yield (Client Connected: {global_weaviate_client.is_connected() if global_weaviate_client else 'None'}) ---"
    )
    yield
    logger.info(
        f"--- Lifespan Shutdown (Client Connected: {global_weaviate_client.is_connected() if global_weaviate_client else 'None'}) ---"
    )

    # --- Watcher Shutdown ---
    logger.info("Lifespan: Stopping active file watchers managed by this instance...")
    active_watcher_names = list(ACTIVE_WATCHERS.keys())
    if active_watcher_names:
        logger.info(f"Found active watchers for codebases: {active_watcher_names}")
        manager = WeaviateManagerBridge(global_weaviate_client)
        for codebase_name_shutdown in active_watcher_names:
            logger.info(f"Lifespan: Stopping watcher for '{codebase_name_shutdown}'...")
            stop_success, stop_msg = stop_watcher(manager, ACTIVE_WATCHERS, codebase_name_shutdown)
            if not stop_success:
                logger.error(
                    f"Lifespan: Error stopping watcher for '{codebase_name_shutdown}': {stop_msg}"
                )
        logger.info("Lifespan: Finished attempting to stop watchers.")
    else:
        logger.info("Lifespan: No active watchers found in this instance.")
    # --- End Watcher Shutdown ---

    logger.info("MCP server shutting down...")
    logger.info("Lifespan: Checking for background tasks to cancel...")
    tasks_to_cancel = list(background_llm_tasks)
    if tasks_to_cancel:
        logger.info(f"Lifespan: Cancelling {len(tasks_to_cancel)} background tasks...")
        for task in tasks_to_cancel:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Lifespan: Background tasks cancelled.")
    else:
        logger.info("Lifespan: No background tasks to cancel.")

    logger.info("Lifespan: Checking Weaviate client connection for closure...")
    logger.warning("Lifespan: Skipping Weaviate client close for diagnostics.")
    logger.info(
        "Lifespan: Weaviate client not connected or doesn't exist, skipping close."
    )
    logger.info("MCP server shutdown complete.")
    logger.info("--- Lifespan End ---")


# --- MCP Server Setup ---
logger.info("--- Attempting FastMCP instantiation ---")

mcp = FastMCP(
    name="code-analysis-mcp",
    lifespan=lifespan,
)
logger.info("--- FastMCP instantiation successful ---")


# --- Helper Functions for Output Processing ---


def _shorten_file_path(
    file_path: str | None, codebase_root_dir: str | None
) -> str | None:
    """Converts an absolute path to a path relative to the codebase root directory."""
    if not file_path or not codebase_root_dir:
        return file_path

    try:
        # Use os.path.relpath for robust, cross-platform path handling
        relative_path = os.path.relpath(file_path, start=codebase_root_dir)

        # If the file is outside the root (starts with ..), return the absolute path
        if relative_path.startswith(".."):
             return os.path.normpath(file_path).replace(os.sep, "/")

        # Standardize separators to forward slashes for API output
        final_path = relative_path.replace(os.sep, "/")
        return final_path if final_path != "." else "./"
    except ValueError:
        # Occurs on Windows if paths are on different drives
        return os.path.normpath(file_path).replace(os.sep, "/")


def _trim_string(value: Any) -> Any:
    """Trims leading/trailing whitespace if the value is a string."""
    if isinstance(value, str):
        return value.strip()
    return value


async def _process_element_properties(
    client: Any,
    tenant_id: str,
    properties: dict[str, Any],
    uuid: str,
    view_type: str = "list",
    codebase_root_dir: str | None = None,
) -> dict[str, Any]:
    """Cleans, filters, and enhances properties for API output, tailored by view_type."""
    logger.debug(
        f"_process_element_properties START for tenant {tenant_id}, uuid {uuid}, view_type {view_type}, root={codebase_root_dir}"  # Updated log
    )
    logger.debug(f"  Input properties: {properties}")

    if not properties:
        logger.debug("_process_element_properties END (empty input)")
        return {}

    processed = {}

    list_view_fields = {"name", "type", "file", "uuid", "description"}
    detail_view_fields = {
        "name",
        "element_type",
        "file_path",
        "start_line",
        "end_line",
        "code_snippet",
        "signature",
        "readable_id",
        "parameters",
        "return_type",
        "decorators",
        "attribute_accesses",
        "base_class_names",
        "llm_description",
        "docstring",
        "user_clarification",
        "uuid",
    }
    processed["uuid"] = uuid
    target_fields = detail_view_fields if view_type == "detail" else list_view_fields

    for key, value in properties.items():
        output_key = key
        if key == "element_type" and "type" in target_fields:
            output_key = "type"
        if key == "file_path" and "file" in target_fields:
            output_key = "file"

        if output_key not in target_fields:
            continue

        processed_value = value
        if output_key == "file":
            processed_value = _shorten_file_path(value, codebase_root_dir)
        elif key in [
            "user_clarification",
            "code_snippet",
            "llm_description",
            "docstring",
        ]:
            processed_value = _trim_string(value)

        if output_key == "description" and view_type == "list":
            llm_desc = _trim_string(properties.get("llm_description", ""))
            docstr = _trim_string(properties.get("docstring", ""))
            processed_value = llm_desc or docstr or None
        elif key in ["llm_description", "docstring"] and view_type == "list":
            continue

        if (
            view_type == "list"
            and not processed_value
            and output_key not in {"name", "type", "file", "uuid"}
        ):
            continue

        processed[output_key] = processed_value

    if view_type == "list" and "description" not in processed:
        processed["description"] = None

    logger.debug(f"  Processed properties: {processed}")
    logger.debug(f"_process_element_properties END for tenant {tenant_id}, uuid {uuid}")
    return processed


def _extract_identifiers(code_snippet: str) -> list[str]:
    """Extracts potential identifiers from a code snippet using AST."""
    identifiers = set()
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
    except SyntaxError:
        logger.warning(
            f"Syntax error parsing code snippet for identifier extraction: {code_snippet[:100]}..."
        )
        return re.findall(r"\b[a-zA-Z_]\w*\b", code_snippet)
    except Exception as e:
        logger.error(f"Error extracting identifiers using AST: {e}")
        return re.findall(r"\b[a-zA-Z_]\w*\b", code_snippet)

    keywords = {
        "if",
        "else",
        "elif",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "def",
        "class",
        "import",
        "from",
        "as",
        "pass",
        "return",
        "yield",
        "lambda",
        "True",
        "False",
        "None",
        "self",
        "cls",
    }
    filtered_identifiers = list(identifiers - keywords)
    logger.debug(f"Extracted identifiers using AST: {filtered_identifiers}")
    return filtered_identifiers


# --- Tool Implementations ---


@mcp.tool(
    name="scan_codebase",
    description="Scans a directory, associates it with a codebase name, analyzes code, uploads results, generates a summary, and automatically starts a file watcher.",
)
async def scan_codebase(
    codebase_name: Annotated[
        str,
        Field(description="User-defined name for the codebase. Must be unique."),
    ],
    directory: Annotated[
        str,
        Field(
            description="Absolute path of the directory containing the source code to scan."
        ),
    ],
):
    """Handles codebase scanning, registry updates, tenant creation, analysis, and summary generation."""
    global ACTIVE_CODEBASE_NAME
    logger.info("--- scan_codebase tool execution START ---")
    logger.info(
        f"Scan Codebase Tool: Received args: codebase_name={codebase_name}, directory={directory}"
    )

    client = global_weaviate_client
    if not client or not client.is_connected():
        logger.error(
            "Scan Codebase Tool: Weaviate client is None or not connected right before use. Aborting."
        )
        return {
            "status": "error",
            "message": "Weaviate client not connected (check at tool execution).",
        }
    else:
        logger.info("Scan Codebase Tool: Client confirmed connected right before use.")

    target_directory = directory

    if not re.match(r"^[a-zA-Z0-9_-]+$", codebase_name):
        return {
            "status": "error",
            "message": "Invalid codebase_name. Use only letters, numbers, underscores, and hyphens.",
        }

    if not os.path.isabs(target_directory):
        logger.error(f"Directory path must be absolute: {target_directory}")
        return {"status": "error", "message": "Directory path must be absolute."}

    abs_directory_path = os.path.normpath(target_directory)

    logger.info(
        f"Scan Codebase Tool: Codebase Name='{codebase_name}', Resolved Directory='{abs_directory_path}'"
    )

    if not os.path.isdir(abs_directory_path):
        logger.error(
            f"Scan Codebase Tool: Directory does not exist: {abs_directory_path}"
        )
        return {
            "status": "error",
            "message": f"Directory not found: {abs_directory_path}",
        }

    tenant_id = codebase_name

    try:
        existing_codebase = get_codebase_details(client, codebase_name)
        if existing_codebase:
            logger.error(
                f"Scan Codebase Tool: Codebase '{codebase_name}' already exists in registry."
            )
            return {
                "status": "error",
                "message": f"Codebase '{codebase_name}' already exists. Use a different name or delete the existing codebase first.",
            }

        code_element_collection = client.collections.get("CodeElement")
        tenant_exists = False
        try:
            tenant_exists = code_element_collection.tenants.exists(tenant_id)
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404:
                logger.info(
                    "CodeElement collection not found during tenant check, proceeding."
                )
            else:
                raise
        except Exception:
            logger.exception("Error checking tenant existence")
            raise

        if tenant_exists:
            logger.error(
                f"Scan Codebase Tool: Tenant '{tenant_id}' already exists but codebase is not in registry. Potential inconsistency."
            )
            return {
                "status": "error",
                "message": f"Data for '{codebase_name}' exists but it's not registered. Consider deleting the codebase ('delete_codebase' tool) and trying again.",
            }

        logger.info(
            f"Scan Codebase Tool: Adding '{codebase_name}' to CodebaseRegistry with status 'Scanning'."
        )
        registry_uuid = add_codebase_registry_entry(
            client, codebase_name, abs_directory_path, "Scanning"
        )
        if not registry_uuid:
            return {
                "status": "error",
                "message": f"Failed to add codebase '{codebase_name}' to registry.",
            }

        logger.info(f"Scan Codebase Tool: Creating tenant '{tenant_id}'.")
        try:
            code_file_collection = client.collections.get("CodeFile")
            if not code_file_collection.tenants.exists(tenant_id):
                code_file_collection.tenants.create([Tenant(name=tenant_id)])
            if not code_element_collection.tenants.exists(tenant_id):
                code_element_collection.tenants.create([Tenant(name=tenant_id)])
            logger.info(
                f"Scan Codebase Tool: Tenant '{tenant_id}' created or already exists."
            )
        except Exception as e:
            logger.exception(
                f"Scan Codebase Tool: Failed to create tenant '{tenant_id}'. Cleaning up registry entry."
            )
            delete_codebase_registry_entry(client, codebase_name)
            return {
                "status": "error",
                "message": f"Failed to create Weaviate tenant '{tenant_id}': {e}",
            }

        logger.info(
            f"Scan Codebase Tool: Starting code scan for '{codebase_name}' in directory '{abs_directory_path}'."
        )
        status_message, processed_uuids = await _scan_cleanup_and_upload(
            client, abs_directory_path, tenant_id=tenant_id
        )

        if "ERROR" in status_message or "failed" in status_message.lower():
            logger.error(
                f"Scan Codebase Tool: Scan failed for codebase '{codebase_name}'. Status: {status_message}. Updating registry to Error."
            )
            update_codebase_registry(client, codebase_name, {"status": "Error"})
            return {
                "status": "error",
                "message": f"Scan failed: {status_message}",
                "codebase_name": codebase_name,
            }

        logger.info(
            f"Scan Codebase Tool: Scan successful for codebase '{codebase_name}'. Updating registry to 'Summarizing'."
        )
        update_codebase_registry(client, codebase_name, {"status": "Summarizing"})

        ACTIVE_CODEBASE_NAME = codebase_name
        logger.info(f"Scan Codebase Tool: Set active codebase to '{codebase_name}'.")

        final_message = f"Scan successful for codebase '{codebase_name}'. Summary generation started. Codebase '{codebase_name}' is now the active codebase."

        summary_task = asyncio.create_task(
            background_generate_summary(client, codebase_name)
        )
        background_llm_tasks.add(summary_task)
        summary_task.add_done_callback(background_llm_tasks.discard)

        llm_tasks_started = 0
        if LLM_ENABLED and processed_uuids:
            llm_tasks_started = len(processed_uuids)
            logger.info(
                f"Scheduling {llm_tasks_started} elements for batch LLM processing in tenant '{tenant_id}'."
            )
            # Use batch processing task
            llm_task = asyncio.create_task(
                _process_llm_batch(client, tenant_id, processed_uuids, refine=True)
            )
            background_llm_tasks.add(llm_task)
            llm_task.add_done_callback(background_llm_tasks.discard)

            final_message += (
                f" Background LLM enrichment started for {llm_tasks_started} elements."
            )
        elif not LLM_ENABLED:
            final_message += " LLM enrichment disabled."

        logger.info(
            f"Scan Codebase Tool: Automatically starting watcher for '{codebase_name}'."
        )
        watcher_started, watcher_msg = await start_watcher_wrapper(codebase_name)
        if watcher_started:
            final_message += f" {watcher_msg}"
            logger.info(
                f"Scan Codebase Tool: Watcher started automatically for '{codebase_name}'."
            )
        else:
            final_message += (
                f" Warning: Failed to automatically start watcher: {watcher_msg}"
            )
            logger.error(
                f"Scan Codebase Tool: Failed to automatically start watcher for '{codebase_name}': {watcher_msg}"
            )

        return {
            "status": "success",
            "message": final_message,
            "codebase_name": codebase_name,
        }

    except Exception as e:
        logger.exception(f"Error during scan_codebase for '{codebase_name}': {e}")
        update_codebase_registry(client, codebase_name, {"status": "Error"})
        return {
            "status": "error",
            "message": f"Scan failed for codebase '{codebase_name}': {e}",
            "codebase_name": codebase_name,
        }


@mcp.tool(
    name="list_codebases",
    description="Lists all registered codebases and their status.",
)
async def list_codebases():
    """Retrieves and returns a list of all codebases from the CodebaseRegistry."""
    logger.info("--- list_codebases tool execution START ---")
    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    try:
        codebases = get_all_codebases(client)
        formatted_codebases = [
            {
                "name": p.get("codebase_name"),
                "directory": p.get("directory"),
                "status": p.get("status"),
                "summary": (
                    p.get("summary", "Not generated yet.")[:150] + "..."
                    if p.get("summary") and len(p.get("summary", "")) > 150
                    else p.get("summary", "Not generated yet.")
                ),
                "dependencies": p.get("dependencies", []),
            }
            for p in codebases
        ]
        logger.info(f"list_codebases: Found {len(formatted_codebases)} codebases.")
        return {"status": "success", "codebases": formatted_codebases}
    except Exception as e:
        logger.exception("Error listing codebases: {e}")
        return {
            "status": "error",
            "message": f"Failed to list codebases: {e}",
        }


@mcp.tool(
    name="select_codebase",
    description="Sets the active codebase context for subsequent commands. Stops the watcher for the previously active codebase, if any.",
)
async def select_codebase(
    codebase_name: Annotated[
        str,
        Field(
            description="Name of the previously scanned codebase to set as the active context."
        ),
    ],
):
    """Sets the global ACTIVE_CODEBASE_NAME."""
    global ACTIVE_CODEBASE_NAME, ACTIVE_WATCHERS
    logger.info("--- select_codebase tool execution START ---")
    new_codebase_name = codebase_name
    old_codebase_name = ACTIVE_CODEBASE_NAME

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    try:
        details = get_codebase_details(client, new_codebase_name)
        if not details:
            return {
                "status": "error",
                "message": f"Codebase '{new_codebase_name}' not found.",
            }

        if old_codebase_name and old_codebase_name != new_codebase_name:
            if old_codebase_name in ACTIVE_WATCHERS:
                logger.info(
                    f"select_codebase: Stopping watcher for previously active codebase '{old_codebase_name}'."
                )
                stop_success, stop_msg = await stop_watcher_wrapper(old_codebase_name)
                if not stop_success:
                    logger.warning(
                        f"select_codebase: Failed to stop watcher for '{old_codebase_name}': {stop_msg}"
                    )
            else:
                logger.info(
                    f"select_codebase: Ensuring watcher registry flag is false for previously active codebase '{old_codebase_name}'."
                )
                update_codebase_registry(
                    client, old_codebase_name, {"watcher_active": False}
                )

        ACTIVE_CODEBASE_NAME = new_codebase_name
        summary = details.get("summary", "No summary available.")
        logger.info(f"select_codebase: Set active codebase to '{new_codebase_name}'.")
        return {
            "status": "success",
            "message": f"Codebase '{new_codebase_name}' selected. Summary: {summary}",
        }
    except Exception as e:
        logger.exception(f"Error selecting codebase '{new_codebase_name}': {e}")
        return {
            "status": "error",
            "message": f"Failed to select codebase '{new_codebase_name}': {e}",
        }


@mcp.tool(
    name="delete_codebase",
    description="Deletes a codebase's analysis data (tenant) and removes it from the registry. Stops the watcher if it was active for the deleted codebase.",
)
async def delete_codebase(
    codebase_name: Annotated[
        str,
        Field(
            description="Name of the codebase whose data (including Weaviate tenant) should be deleted."
        ),
    ],
):
    """Stops watcher (if active), deletes the Weaviate tenant and the CodebaseRegistry entry."""
    global ACTIVE_CODEBASE_NAME
    logger.info("--- delete_codebase tool execution START ---")
    tenant_id = codebase_name

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    logger.warning(
        f"Attempting to delete codebase '{codebase_name}' (tenant '{tenant_id}')."
    )
    try:
        logger.info(
            f"Attempting to stop watcher for codebase '{codebase_name}' before deletion."
        )
        stop_success, stop_msg = await stop_watcher_wrapper(codebase_name)
        if not stop_success:
            logger.warning(
                f"Could not definitively stop watcher for '{codebase_name}': {stop_msg}. Proceeding with deletion."
            )
        else:
            logger.info(f"Watcher stop signal sent/confirmed for '{codebase_name}'.")

        logger.info(f"Deleting tenant '{tenant_id}' data...")
        tenant_deleted = delete_tenant(client, tenant_id)
        if not tenant_deleted:
            logger.error(
                f"Failed to fully delete Weaviate tenant '{tenant_id}', but proceeding with registry deletion."
            )

        logger.info(f"Deleting codebase '{codebase_name}' from registry...")
        registry_deleted = delete_codebase_registry_entry(client, codebase_name)
        if not registry_deleted:
            logger.error(f"Failed to delete codebase '{codebase_name}' from registry.")
            return {
                "status": "error",
                "message": f"Tenant data deleted, but failed to remove '{codebase_name}' from registry.",
            }

        if ACTIVE_CODEBASE_NAME == codebase_name:
            ACTIVE_CODEBASE_NAME = None
            logger.info(f"Cleared active codebase as '{codebase_name}' was deleted.")

        logger.info(f"Successfully deleted codebase '{codebase_name}'.")
        return {
            "status": "success",
            "message": f"Codebase '{codebase_name}' deleted successfully.",
        }

    except Exception as e:
        logger.exception(f"Error deleting codebase '{codebase_name}': {e}")
        return {
            "status": "error",
            "message": f"Failed to delete codebase '{codebase_name}': {e}",
        }


@mcp.tool(
    name="find_element",
    description="Finds code elements in the active codebase based on criteria.",
)
async def find_element(
    name: Annotated[
        Optional[str],
        Field(
            description="Name of the code element (e.g., function name, class name) to search for."
        ),
    ] = None,
    file_path: Annotated[
        Optional[str],
        Field(
            description="File path where the element is defined (relative to codebase root, e.g., 'src/my_module.py')."
        ),
    ] = None,
    element_type: Annotated[
        Optional[str],
        Field(
            description="Type of the element to search for (e.g., 'function', 'class', 'import')."
        ),
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of matching elements to return.")
    ] = 5,
    include_dependencies: Annotated[
        bool, Field(description="Whether to include results from dependency codebases.")
    ] = False,
):
    """Finds code elements using filters within the active codebase and optionally its dependencies."""
    logger.info("--- find_element tool execution START ---")
    if not ACTIVE_CODEBASE_NAME:
        return {
            "status": "error",
            "message": "No active codebase selected. Use 'select_codebase' first.",
        }

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    primary_tenant_id = ACTIVE_CODEBASE_NAME
    tenant_ids_to_query = [primary_tenant_id]

    # Fetch dependencies if requested
    if include_dependencies:
        logger.info(
            f"Including dependencies for codebase '{primary_tenant_id}' in find_element query."
        )
        codebase_details = get_codebase_details(client, primary_tenant_id)
        if codebase_details and codebase_details.get("dependencies"):
            dependencies = codebase_details["dependencies"]
            logger.info(f"Found dependencies: {dependencies}")
            # Check if dependency tenants exist before adding
            elements_collection = client.collections.get("CodeElement")
            for dep_name in dependencies:
                if elements_collection.tenants.exists(dep_name):
                    tenant_ids_to_query.append(dep_name)
                else:
                    logger.warning(
                        f"Dependency codebase (tenant) '{dep_name}' not found in Weaviate. Skipping."
                    )
        else:
            logger.info(
                f"No dependencies found or defined for codebase '{primary_tenant_id}'."
            )

    logger.info(
        f"Finding elements across tenants {tenant_ids_to_query} with args: name={name}, file_path={file_path}, element_type={element_type}, limit={limit}"
    )
    try:
        # Check existence of primary tenant (already done implicitly by get_codebase_details if include_dependencies=True)
        if not include_dependencies:
            if not client.collections.get("CodeElement").tenants.exists(
                primary_tenant_id
            ):
                return {
                    "status": "error",
                    "message": f"Active codebase '{primary_tenant_id}' data not found (tenant missing).",
                }

        abs_file_path = None
        if file_path:
            if os.path.isabs(file_path):
                abs_file_path = os.path.normpath(file_path)
            else:
                # Need codebase directory to resolve relative path - use primary codebase for this
                codebase_details = get_codebase_details(client, primary_tenant_id)
                if codebase_details and codebase_details.get("directory"):
                    codebase_root_dir = codebase_details["directory"]
                    abs_file_path = os.path.abspath(
                        os.path.join(codebase_root_dir, file_path)
                    )
                else:
                    logger.warning(
                        f"Could not resolve relative file path '{file_path}' as codebase directory is unknown for '{primary_tenant_id}'. Searching without file path filter."
                    )
            if abs_file_path:
                logger.debug(
                    f"find_element: Converted file path '{file_path}' to absolute '{abs_file_path}' for query."
                )

        # Call the multi-tenant version of find_element_by_name
        results = find_element_by_name(
            client,
            tenant_ids_to_query,
            element_name=name,
            file_path=abs_file_path,
            element_type=element_type,
            limit=limit,
        )
        processed_results = []
        # Need to fetch codebase details again for path shortening if dependencies were included
        codebase_details_map = {}
        if include_dependencies:
            all_details = await asyncio.gather(
                *[
                    asyncio.to_thread(get_codebase_details, client, tid)
                    for tid in tenant_ids_to_query
                ]
            )
            for details in all_details:
                if details:
                    codebase_details_map[details["codebase_name"]] = details

        for r in results:
            result_tenant_id = r.properties.pop("_tenant_id", primary_tenant_id)
            # Get the correct root dir for path shortening
            current_codebase_details = codebase_details_map.get(result_tenant_id)
            codebase_root = (
                current_codebase_details.get("directory")
                if current_codebase_details
                else None
            )
            processed_props = await _process_element_properties(
                client,
                result_tenant_id,
                r.properties,
                str(r.uuid),
                view_type="list",
                codebase_root_dir=codebase_root,
            )
            # Optionally add codebase name to output if multiple tenants were queried
            if len(tenant_ids_to_query) > 1:
                processed_props["codebase"] = result_tenant_id
            processed_results.append(processed_props)

        return {
            "status": "success",
            "count": len(processed_results),
            "elements": processed_results,
        }
    except Exception as e:
        logger.exception(
            f"Error finding elements across codebases {tenant_ids_to_query}: {e}"
        )
        return {
            "status": "error",
            "message": f"Failed to find elements across codebases {tenant_ids_to_query}: {e}",
        }


@mcp.tool(
    name="get_details",
    description="Retrieves detailed information for a specific code element in the active codebase.",
)
async def get_details(
    uuid: Annotated[
        str,
        Field(
            description="The unique identifier (UUID) of the specific code element to retrieve details for."
        ),
    ],
):
    """Gets details for a specific element UUID within the ACTIVE_CODEBASE_NAME."""
    logger.info("--- get_details tool execution START ---")
    if not ACTIVE_CODEBASE_NAME:
        return {
            "status": "error",
            "message": "No active codebase selected. Use 'select_codebase' first.",
        }

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    tenant_id = ACTIVE_CODEBASE_NAME
    logger.info(
        f"Getting details for element '{uuid}' in active codebase '{tenant_id}'."
    )
    try:
        if not client.collections.get("CodeElement").tenants.exists(tenant_id):
            return {
                "status": "error",
                "message": f"Active codebase '{tenant_id}' data not found (tenant missing).",
            }

        details = get_element_details(client, tenant_id, uuid)
        if not details:
            return {
                "status": "not_found",
                "message": f"Element with UUID '{uuid}' not found in codebase '{tenant_id}'.",
            }

        # Fetch codebase root for path shortening
        codebase_details = get_codebase_details(client, tenant_id)
        codebase_root = codebase_details.get("directory") if codebase_details else None
        processed_details = await _process_element_properties(
            client,
            tenant_id,
            details.properties,
            uuid,
            view_type="detail",
            codebase_root_dir=codebase_root,
        )
        return {"status": "success", "details": processed_details}
    except Exception as e:
        logger.exception(
            f"Error getting details for element '{uuid}' in codebase '{tenant_id}': {e}"
        )
        return {
            "status": "error",
            "message": f"Failed to get details for element '{uuid}' in codebase '{tenant_id}': {e}",
        }


@mcp.tool(
    name="analyze_snippet",
    description="Analyzes a code snippet to find related elements in the active codebase.",
)
async def analyze_snippet(
    code_snippet: Annotated[
        str,
        Field(
            description="A snippet of Python code to analyze for finding related elements within the active codebase."
        ),
    ],
):
    """Analyzes a snippet, finds identifiers, and searches within the ACTIVE_CODEBASE_NAME."""
    logger.info("--- analyze_snippet tool execution START ---")
    if not ACTIVE_CODEBASE_NAME:
        return {
            "status": "error",
            "message": "No active codebase selected. Use 'select_codebase' first.",
        }

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    tenant_id = ACTIVE_CODEBASE_NAME
    logger.info(
        f"Analyzing snippet in active codebase '{tenant_id}': '{code_snippet[:50]}...'"
    )
    try:
        if not client.collections.get("CodeElement").tenants.exists(tenant_id):
            return {
                "status": "error",
                "message": f"Active codebase '{tenant_id}' data not found (tenant missing).",
            }

        identifiers = _extract_identifiers(code_snippet)
        if not identifiers:
            return {
                "status": "success",
                "message": "No relevant identifiers found in the snippet.",
                "related_elements": [],
            }

        logger.debug(f"Found identifiers: {identifiers}")

        unique_elements = {}
        errors = []
        for ident in identifiers:
            try:
                # Call find_element directly with keyword arguments
                result_data = await find_element(name=ident, limit=3)

                if result_data.get("status") == "success":
                    for element_summary in result_data.get("elements", []):
                        element_uuid = element_summary.get("uuid")
                        if element_uuid and element_uuid not in unique_elements:
                            unique_elements[element_uuid] = element_summary
                elif result_data.get("status") != "error":
                    errors.append(
                        f"Error searching for identifier '{ident}': {result_data.get('message')}"
                    )

            except Exception as search_e:
                logger.exception(
                    f"Analyze Snippet: Error processing identifier '{ident}': {search_e}"
                )
                errors.append(f"Error processing identifier '{ident}': {search_e}")

        final_elements = list(unique_elements.values())
        message = f"Found {len(final_elements)} potentially related unique elements for identifiers: {', '.join(identifiers)}."
        if errors:
            message += f" Encountered errors: {'; '.join(errors)}"

        return {
            "status": "success",
            "message": message,
            "related_elements": final_elements,
        }
    except Exception as e:
        logger.exception(f"Error analyzing snippet in codebase '{tenant_id}': {e}")
        return {
            "status": "error",
            "message": f"Internal server error while analyzing snippet in codebase '{tenant_id}'. Check server logs.",
        }


@mcp.tool(
    name="ask_question",
    description="Answers a question about the active codebase using RAG.",
)
async def ask_question(
    query: Annotated[
        str,
        Field(
            description="Natural language question about the codebase of the currently active codebase."
        ),
    ],
):
    """Answers a question using RAG against the ACTIVE_CODEBASE_NAME."""
    logger.info("--- ask_question tool execution START ---")
    if not ACTIVE_CODEBASE_NAME:
        return {
            "status": "error",
            "message": "No active codebase selected. Use 'select_codebase' first.",
        }

    client = global_weaviate_client

    tenant_id = ACTIVE_CODEBASE_NAME
    logger.info(f"Asking question about active codebase '{tenant_id}': '{query}'")
    try:
        answer = await answer_codebase_question(
            query, client=client, tenant_id=tenant_id
        )
        if answer.startswith("ERROR:"):
            return {"status": "error", "message": answer}
        else:
            return {"status": "success", "answer": answer}
    except Exception as e:
        logger.exception(f"Error asking question about codebase '{tenant_id}': {e}")
        return {
            "status": "error",
            "message": f"Failed to answer question about codebase '{tenant_id}': {e}",
        }


@mcp.tool(
    name="trigger_llm_processing",
    description="Triggers background LLM processing for code elements in the active codebase.",
)
async def trigger_llm_processing(
    uuids: Annotated[
        Optional[List[str]],
        Field(
            description="A specific list of element UUIDs to queue for background LLM description generation/refinement."
        ),
    ] = None,
    rerun_all: Annotated[
        bool,
        Field(
            description="If true, queue all elements in the active codebase for LLM processing."
        ),
    ] = False,
    skip_enriched: Annotated[
        bool,
        Field(
            description="If true, skip processing for elements that already have an LLM-generated description."
        ),
    ] = True,
):
    """Triggers LLM enrichment/refinement for the ACTIVE_CODEBASE_NAME."""
    logger.info("--- trigger_llm_processing tool execution START ---")
    if not ACTIVE_CODEBASE_NAME:
        return {
            "status": "error",
            "message": "No active codebase selected. Use 'select_codebase' first.",
        }
    if not LLM_ENABLED:
        return {"status": "error", "message": "LLM processing is disabled."}

    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    tenant_id = ACTIVE_CODEBASE_NAME
    logger.info(
        f"Triggering LLM processing in active codebase '{tenant_id}' with args: uuids={uuids}, rerun_all={rerun_all}, skip_enriched={skip_enriched}"
    )
    try:
        element_collection = client.collections.get("CodeElement")
        if not element_collection.tenants.exists(tenant_id):
            return {
                "status": "error",
                "message": f"Active codebase '{tenant_id}' data not found (tenant missing).",
            }

        uuids_to_process = []
        if rerun_all:
            logger.info(f"Fetching all elements for codebase '{tenant_id}'...")
            response = element_collection.with_tenant(tenant_id).query.fetch_objects(
                limit=10000
            )
            all_uuids = [str(obj.uuid) for obj in response.objects]
            logger.info(f"Found {len(all_uuids)} total elements.")
            if skip_enriched:
                logger.info("Filtering elements to skip already enriched ones...")
                detail_tasks = [
                    asyncio.to_thread(get_element_details, client, tenant_id, uuid)
                    for uuid in all_uuids
                ]
                details_list = await asyncio.gather(
                    *detail_tasks, return_exceptions=True
                )

                for i, detail_result in enumerate(details_list):
                    uuid_item = all_uuids[i]
                    if isinstance(detail_result, Exception):
                        logger.error(
                            f"Error fetching details for {uuid_item} during filtering: {detail_result}"
                        )
                        uuids_to_process.append(uuid_item)
                    elif detail_result:
                        props = detail_result.properties
                        desc = props.get("llm_description", "")
                        if not desc or desc == "[Description not generated]":
                            uuids_to_process.append(uuid_item)
                    else:
                        logger.warning(
                            f"Element {uuid_item} not found during filtering. Skipping."
                        )
            else:
                uuids_to_process = all_uuids

        elif uuids:
            logger.info(f"Processing specific UUIDs: {uuids}")
            if skip_enriched:
                logger.info("Filtering specific UUIDs to skip already enriched ones...")
                detail_tasks = [
                    asyncio.to_thread(get_element_details, client, tenant_id, uuid_item)
                    for uuid_item in uuids
                ]
                details_list = await asyncio.gather(
                    *detail_tasks, return_exceptions=True
                )
                for i, detail_result in enumerate(details_list):
                    uuid_item = uuids[i]
                    if isinstance(detail_result, Exception):
                        logger.error(
                            f"Error fetching details for {uuid_item} during filtering: {detail_result}"
                        )
                        uuids_to_process.append(uuid_item)
                    elif detail_result:
                        props = detail_result.properties
                        desc = props.get("llm_description", "")
                        if not desc or desc == "[Description not generated]":
                            uuids_to_process.append(uuid_item)
                    else:
                        logger.warning(
                            f"Specified element {uuid_item} not found during filtering. Skipping."
                        )
            else:
                uuids_to_process = uuids
        else:
            return {
                "status": "error",
                "message": "Must provide either 'uuids' list or set 'rerun_all' to true.",
            }

        if not uuids_to_process:
            return {
                "status": "success",
                "message": f"No elements found to process in codebase '{tenant_id}' based on criteria.",
            }

        logger.info(
            f"Scheduling LLM processing for {len(uuids_to_process)} elements in codebase '{tenant_id}'."
        )

        # Use batch processing task
        count = len(uuids_to_process)
        task = asyncio.create_task(
            _process_llm_batch(client, tenant_id, uuids_to_process, refine=True)
        )
        background_llm_tasks.add(task)
        task.add_done_callback(background_llm_tasks.discard)

        return {
            "status": "success",
            "message": f"Background batch LLM processing triggered for {count} elements in codebase '{tenant_id}'.",
        }
    except Exception as e:
        logger.exception(
            f"Error triggering LLM processing in codebase '{tenant_id}': {e}"
        )
        return {
            "status": "error",
            "message": f"Failed to trigger LLM processing in codebase '{tenant_id}': {e}",
        }


# --- Watcher Tools ---
@mcp.tool(
    name="start_watcher",
    description="Starts a file watcher for the specified codebase directory.",
)
async def start_watcher_tool(
    codebase_name: Annotated[
        str,
        Field(
            description="Name of the codebase for which to start or stop the file watcher."
        ),
    ],
):
    """MCP tool to start the file watcher."""
    success, message = await start_watcher_wrapper(codebase_name)
    return {"status": "success" if success else "error", "message": message}


@mcp.tool(
    name="stop_watcher",
    description="Stops the file watcher for the specified codebase.",
)
async def stop_watcher_tool(
    codebase_name: Annotated[
        str,
        Field(
            description="Name of the codebase for which to start or stop the file watcher."
        ),
    ],
):
    """MCP tool to stop the file watcher."""
    success, message = await stop_watcher_wrapper(codebase_name)
    return {"status": "success" if success else "error", "message": message}


# --- Dependency Management Tools ---
@mcp.tool(
    name="add_codebase_dependency",
    description="Adds a dependency relationship between two codebases.",
)
async def add_codebase_dependency(
    codebase_name: Annotated[
        str, Field(description="The name of the codebase that has the dependency.")
    ],
    dependency_name: Annotated[
        str, Field(description="The name of the codebase it depends on.")
    ],
):
    """Adds a dependency entry to the CodebaseRegistry."""
    logger.info("--- add_codebase_dependency tool execution START ---")
    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    codebase_details = get_codebase_details(client, codebase_name)
    dependency_details = get_codebase_details(client, dependency_name)

    if not codebase_details:
        return {"status": "error", "message": f"Codebase '{codebase_name}' not found."}
    if not dependency_details:
        return {
            "status": "error",
            "message": f"Dependency codebase '{dependency_name}' not found.",
        }

    current_deps = codebase_details.get("dependencies") or []
    if dependency_name not in current_deps:
        new_deps = current_deps + [dependency_name]
        update_success = update_codebase_registry(
            client, codebase_name, {"dependencies": new_deps}
        )
        if update_success:
            return {
                "status": "success",
                "message": f"Added dependency '{dependency_name}' to codebase '{codebase_name}'.",
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to update dependencies for codebase '{codebase_name}'.",
            }
    else:
        return {
            "status": "success",
            "message": f"Codebase '{codebase_name}' already depends on '{dependency_name}'.",
        }


@mcp.tool(
    name="remove_codebase_dependency",
    description="Removes a dependency relationship between two codebases.",
)
async def remove_codebase_dependency(
    codebase_name: Annotated[
        str,
        Field(description="The name of the codebase to remove the dependency from."),
    ],
    dependency_name: Annotated[
        str, Field(description="The name of the dependency codebase to remove.")
    ],
):
    """Removes a dependency entry from the CodebaseRegistry."""
    logger.info("--- remove_codebase_dependency tool execution START ---")
    client = global_weaviate_client
    if not client or not client.is_connected():
        return {"status": "error", "message": "Weaviate client not connected."}

    codebase_details = get_codebase_details(client, codebase_name)
    if not codebase_details:
        return {"status": "error", "message": f"Codebase '{codebase_name}' not found."}

    current_deps = codebase_details.get("dependencies") or []
    if dependency_name in current_deps:
        new_deps = [dep for dep in current_deps if dep != dependency_name]
        update_success = update_codebase_registry(
            client, codebase_name, {"dependencies": new_deps}
        )
        if update_success:
            return {
                "status": "success",
                "message": f"Removed dependency '{dependency_name}' from codebase '{codebase_name}'.",
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to update dependencies for codebase '{codebase_name}'.",
            }
    else:
        return {
            "status": "success",
            "message": f"Codebase '{codebase_name}' does not depend on '{dependency_name}'.",
        }


# --- Run the server ---
if __name__ == "__main__":
    logger.info("Attempting to start the MCP server with mcp.run()...")
    try:
        logger.info("Calling mcp.run()...")
        mcp.run()
        logger.info("mcp.run() finished.")
    except AttributeError:
        logger.error(
            "mcp.run() method not found. Start manually (e.g., using uvicorn)."
        )
    except Exception as run_e:
        logger.exception(
            f"An error occurred while trying to run the MCP server: {run_e}"
        )
    except Exception as main_e:
        logger.exception(f"Error during mcp.run(): {main_e}")
