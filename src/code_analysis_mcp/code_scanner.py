import os
import ast
import google.generativeai as genai
import google.api_core.exceptions
import datetime
import asyncio
import re
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from weaviate_client import (  # Use relative import
    create_weaviate_client,
    get_all_code_files,
    get_code_files_metadata,
    delete_elements_by_file_path,
    delete_code_file,
    add_objects_batch,
    add_references_batch,
    find_element_by_name,
    get_element_details,
    update_element_properties,
)
from weaviate.util import generate_uuid5
import logging
import aiofiles
import atexit

logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded for code_scanner.")

# --- Shared Executor ---
_executor = None

def get_shared_executor():
    """Returns a shared ProcessPoolExecutor."""
    global _executor
    if _executor is None:
        max_workers = os.cpu_count() or 4
        _executor = ProcessPoolExecutor(max_workers=max_workers)
    return _executor

def _shutdown_executor():
    """Clean shutdown for the shared executor."""
    global _executor
    if _executor:
        _executor.shutdown()

atexit.register(_shutdown_executor)


# --- Configuration ---
VERBOSE_LOGGING = os.getenv("CODE_SCANNER_VERBOSE", "False").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATE_LLM_DESCRIPTIONS = (
    os.getenv("GENERATE_LLM_DESCRIPTIONS", "false").lower() == "true"
)
model = None
embedding_model_name = None


if not GEMINI_API_KEY:
    if VERBOSE_LOGGING:
        logger.warning("GEMINI_API_KEY not found. LLM features disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GENERATION_MODEL_NAME = os.getenv(
            "GENERATION_MODEL_NAME", "models/gemini-2.0-flash-001"
        )
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
        if VERBOSE_LOGGING:
            logger.info(f"Using Generation Model: {GENERATION_MODEL_NAME}")
            logger.info(f"Using Embedding Model: {embedding_model_name}")
    except Exception as e:
        logger.error(f"Error initializing Gemini models: {type(e).__name__}: {e}")
        raise


# --- Helper Functions ---
def _find_python_files_sync(directory):
    """Sync helper to find python files with basic filtering."""
    assert os.path.isabs(directory), "Directory must be absolute."

    python_files = []
    start_dir = os.path.abspath(directory)

    # Basic filtering for common non-source directories
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist'}

    for root, dirs, files in os.walk(start_dir):
        # Modify dirs in-place to skip traversing ignored directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

        for f in files:
            if f.endswith(".py"):
                python_files.append(os.path.join(root, f))

    return python_files

async def find_python_files(directory):
    """Finds all python files recursively in the given directory asynchronously."""
    return await asyncio.to_thread(_find_python_files_sync, directory)


async def read_file_content(abs_path):
    """Reads the content of a file at the given absolute path asynchronously."""
    try:
        async with aiofiles.open(abs_path, "r", encoding="utf-8") as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading file {abs_path}: {e}")
        return None

def parse_code(file_content, file_path, verbose=VERBOSE_LOGGING):
    """Parses code using AST. file_path is used for error reporting."""
    try:
        return ast.parse(file_content, filename=file_path)
    except SyntaxError as e:
        if verbose:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
        return None

def process_file_worker(file_path_rel, file_content, file_uuid, file_mtime):
    """
    Worker function to parse code and visit nodes.
    Run in a separate process to avoid blocking the event loop with CPU-bound AST operations.
    """
    try:
        parsed_code = parse_code(file_content, file_path_rel, VERBOSE_LOGGING)
        if parsed_code is None:
            return [], []

        # Initialize visitor with explicit mtime
        visitor = CodeVisitor(file_path_rel, file_content, file_uuid, file_mtime=file_mtime)
        visitor.visit(parsed_code)

        return visitor.elements, visitor.references
    except Exception as e:
        # Logging in a worker process might need configuration to show up in main logs
        # For now, print to stderr or just return empty
        print(f"Worker failed for {file_path_rel}: {e}")
        return [], []


# --- AST Visitor (Structural Analysis Only) ---
class CodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path, code_content, file_uuid, file_mtime=None):
        """Initializes the visitor."""
        self.file_path = file_path
        self.code_content = code_content
        self.file_uuid = file_uuid
        self.elements = []
        self.references = []
        self.element_uuid_map = {}
        self.scope_stack = []
        self.current_attributes = []

        if file_mtime:
            self.file_mtime = file_mtime
        else:
            # Fallback (should generally be provided)
            self.file_mtime = datetime.datetime.now(datetime.timezone.utc)

    def _get_element_key(self, element_type, name, start_line):
        """Generates a consistent key for an element."""
        return f"{self.file_path}:{element_type}:{name}:{start_line}"

    def _generate_uuid(self, key):
        """Generates a UUID5 based on the element key."""
        return generate_uuid5(key)

    def _get_parent_scope_uuid(self):
        """Gets the UUID of the immediate parent scope (function/class)."""
        return self.scope_stack[-1][0] if self.scope_stack else None

    def visit_FunctionDef(self, node):
        """Visits FunctionDef nodes (functions and methods)."""
        func_key = self._get_element_key("function", node.name, node.lineno)
        func_uuid = self._generate_uuid(func_key)
        self.element_uuid_map[func_key] = func_uuid
        parent_scope_uuid = self._get_parent_scope_uuid()
        readable_id = func_key
        decorators = [ast.unparse(d) for d in node.decorator_list]

        params = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                try:
                    annotation_src = ast.get_source_segment(
                        self.code_content, arg.annotation
                    )
                    param_str += f": {annotation_src or ast.unparse(arg.annotation)}"
                except Exception:
                    param_str += ": ?"
            params.append(param_str)
        signature = f"{node.name}({', '.join(params)})"

        return_type_str = ""
        if node.returns:
            try:
                return_src = ast.get_source_segment(self.code_content, node.returns)
                return_type_str = return_src or ast.unparse(node.returns)
            except Exception:
                return_type_str = "?"
            signature += f" -> {return_type_str}"

        original_attributes = self.current_attributes
        self.current_attributes = []

        element_props = {
            "name": node.name,
            "element_type": "function",
            "file_path": self.file_path,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "docstring": ast.get_docstring(node) or "",
            "code_snippet": ast.get_source_segment(self.code_content, node) or "",
            "parameters": params,
            "return_type": return_type_str,
            "signature": signature,
            "readable_id": readable_id,
            "decorators": decorators,
            "parent_scope_uuid": parent_scope_uuid,
            "user_clarification": "",
            "llm_description": "[Description not generated]",
            "last_modified": self.file_mtime,
        }
        element_data = {
            "uuid": func_uuid,
            "properties": element_props,
            "vector": None,
        }

        self.scope_stack.append((func_uuid, "function"))
        self.generic_visit(node)
        self.scope_stack.pop()

        element_props["attribute_accesses"] = self.current_attributes
        self.current_attributes = original_attributes

        self.elements.append(element_data)

        self.references.append(
            {
                "from_uuid": func_uuid,
                "from_collection": "CodeElement",
                "from_property": "defined_in_file",
                "to": self.file_uuid,
            }
        )
        if (
            parent_scope_uuid
            and self.scope_stack
            and self.scope_stack[-1][1] == "class"
        ):
            self.references.append(
                {
                    "from_uuid": parent_scope_uuid,
                    "from_collection": "CodeElement",
                    "from_property": "defines_method",
                    "to": func_uuid,
                }
            )
            self.references.append(
                {
                    "from_uuid": func_uuid,
                    "from_collection": "CodeElement",
                    "from_property": "method_of",
                    "to": parent_scope_uuid,
                }
            )

    def visit_ClassDef(self, node):
        """Visits ClassDef nodes."""
        class_key = self._get_element_key("class", node.name, node.lineno)
        class_uuid = self._generate_uuid(class_key)
        self.element_uuid_map[class_key] = class_uuid
        parent_scope_uuid = self._get_parent_scope_uuid()
        readable_id = class_key
        decorators = [ast.unparse(d) for d in node.decorator_list]
        base_class_names = [ast.unparse(b) for b in node.bases]

        element_props = {
            "name": node.name,
            "element_type": "class",
            "file_path": self.file_path,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "docstring": ast.get_docstring(node) or "",
            "code_snippet": ast.get_source_segment(self.code_content, node) or "",
            "readable_id": readable_id,
            "decorators": decorators,
            "parent_scope_uuid": parent_scope_uuid,
            "base_class_names": base_class_names,
            "user_clarification": "",
            "llm_description": "[Description not generated]",
            "last_modified": self.file_mtime,
        }
        element_data = {
            "uuid": class_uuid,
            "properties": element_props,
            "vector": None,
        }

        self.scope_stack.append((class_uuid, "class"))
        self.generic_visit(node)
        self.scope_stack.pop()

        self.elements.append(element_data)

        self.references.append(
            {
                "from_uuid": class_uuid,
                "from_collection": "CodeElement",
                "from_property": "defined_in_file",
                "to": self.file_uuid,
            }
        )

    def visit_Import(self, node):
        """Visits Import nodes."""
        for alias in node.names:
            import_key = self._get_element_key("import", alias.name, node.lineno)
            import_uuid = self._generate_uuid(import_key)
            self.element_uuid_map[import_key] = import_uuid
            element_data = {
                "uuid": import_uuid,
                "properties": {
                    "name": alias.name
                    + (f" as {alias.asname}" if alias.asname else ""),
                    "element_type": "import",
                    "file_path": self.file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "code_snippet": ast.get_source_segment(self.code_content, node)
                    or "",
                    "readable_id": import_key,
                    "parent_scope_uuid": self._get_parent_scope_uuid(),
                    "llm_description": "",
                    "user_clarification": "",
                    "docstring": "",
                    "last_modified": self.file_mtime,
                },
            }
            self.elements.append(element_data)
            self.references.append(
                {
                    "from_uuid": import_uuid,
                    "from_collection": "CodeElement",
                    "from_property": "defined_in_file",
                    "to": self.file_uuid,
                }
            )

    def visit_ImportFrom(self, node):
        """Visits ImportFrom nodes."""
        module_name = node.module or ""
        for alias in node.names:
            full_name = f"{module_name}.{alias.name}"
            import_key = self._get_element_key("import_from", full_name, node.lineno)
            import_uuid = self._generate_uuid(import_key)
            self.element_uuid_map[import_key] = import_uuid
            element_data = {
                "uuid": import_uuid,
                "properties": {
                    "name": full_name + (f" as {alias.asname}" if alias.asname else ""),
                    "element_type": "import_from",
                    "file_path": self.file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "code_snippet": ast.get_source_segment(self.code_content, node)
                    or "",
                    "readable_id": import_key,
                    "parent_scope_uuid": self._get_parent_scope_uuid(),
                    "llm_description": "",
                    "user_clarification": "",
                    "docstring": "",
                    "last_modified": self.file_mtime,
                },
            }
            self.elements.append(element_data)
            self.references.append(
                {
                    "from_uuid": import_uuid,
                    "from_collection": "CodeElement",
                    "from_property": "defined_in_file",
                    "to": self.file_uuid,
                }
            )

    def visit_Call(self, node):
        """Visits Call nodes to identify function/method calls."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            call_key = self._get_element_key("call", func_name, node.lineno)
            call_uuid = self._generate_uuid(call_key)
            parent_scope_uuid = self._get_parent_scope_uuid()

            element_data = {
                "uuid": call_uuid,
                "properties": {
                    "name": func_name,
                    "element_type": "call",
                    "file_path": self.file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "code_snippet": ast.get_source_segment(self.code_content, node)
                    or "",
                    "readable_id": call_key,
                    "parent_scope_uuid": parent_scope_uuid,
                    "llm_description": "",
                    "user_clarification": "",
                    "docstring": "",
                    "last_modified": self.file_mtime,
                },
            }
            self.elements.append(element_data)

            self.references.append(
                {
                    "from_uuid": call_uuid,
                    "from_collection": "CodeElement",
                    "from_property": "defined_in_file",
                    "to": self.file_uuid,
                }
            )

            if parent_scope_uuid:
                self.references.append(
                    {
                        "from_uuid": parent_scope_uuid,
                        "from_collection": "CodeElement",
                        "from_property": "calls_function",
                        "to": call_uuid,
                    }
                )

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visits Assign nodes to identify variable assignments."""
        for target in node.targets:
            element_data = None
            if isinstance(target, ast.Name):
                assign_key = self._get_element_key("assignment", target.id, node.lineno)
                assign_uuid = self._generate_uuid(assign_key)
                self.element_uuid_map[assign_key] = assign_uuid
                parent_scope_uuid = self._get_parent_scope_uuid()

                element_data = {
                    "uuid": assign_uuid,
                    "properties": {
                        "name": target.id,
                        "element_type": "variable_assignment",
                        "file_path": self.file_path,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "code_snippet": ast.get_source_segment(self.code_content, node)
                        or "",
                        "readable_id": assign_key,
                        "parent_scope_uuid": parent_scope_uuid,
                        "llm_description": "",
                        "user_clarification": "",
                        "docstring": "",
                        "last_modified": self.file_mtime,
                    },
                }
                if element_data:
                    self.elements.append(element_data)
                    self.references.append(
                        {
                            "from_uuid": assign_uuid,
                            "from_collection": "CodeElement",
                            "from_property": "defined_in_file",
                            "to": self.file_uuid,
                        }
                    )
                    parent_scope_uuid_for_ref = self._get_parent_scope_uuid()
                    if parent_scope_uuid_for_ref:
                        self.references.append(
                            {
                                "from_uuid": parent_scope_uuid_for_ref,
                                "from_collection": "CodeElement",
                                "from_property": "defines_variable",
                                "to": assign_uuid,
                            }
                        )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visits Attribute nodes to track attribute accesses within scopes."""
        if self.scope_stack and self.scope_stack[-1][1] in ["function", "class"]:
            try:
                attr_str = ast.unparse(node)
                if attr_str not in self.current_attributes:
                    self.current_attributes.append(attr_str)
            except Exception:
                logger.warning(
                    f"Could not unparse attribute node at {self.file_path}:{node.lineno}"
                )
        self.generic_visit(node)


# --- Analysis Functions ---


def _extract_identifiers(code_snippet: str) -> list[str]:
    """Extracts potential identifiers from a code snippet.

    This function uses a regular expression to find potential identifiers
    (variable names, function names, etc.) within a given code snippet.
    It filters out common keywords and built-in names to reduce noise.
    """
    try:
        identifiers = re.findall(r"\b[a-zA-Z_]\w*\b", code_snippet)
        keywords = [
            "if",
            "else",
            "for",
            "while",
            "return",
            "def",
            "class",
            "import",
            "from",
        ]
        identifiers = [i for i in identifiers if i not in keywords]
        logger.debug(f"Extracted identifiers: {identifiers}")
        return identifiers
    except Exception as e:
        logger.error(f"Error extracting identifiers: {type(e).__name__}: {e}")
        return []


async def enrich_element(client, tenant_id: str, element_uuid: str) -> bool:
    """Wrapper for single element enrichment using the batch logic."""
    result = await enrich_elements_batch(client, tenant_id, [element_uuid])
    return result.get("success_count", 0) > 0

async def enrich_elements_batch(client, tenant_id: str, element_uuids: list[str]) -> dict:
    """
    Fetches elements in batch, generates LLM descriptions and embeddings in parallel,
    and updates them in Weaviate using batch operations.
    """
    if not GENERATE_LLM_DESCRIPTIONS or not model or not embedding_model_name:
        logger.debug(f"LLM generation disabled, skipping enrichment for {len(element_uuids)} elements.")
        return {"success_count": 0, "failed_count": 0}

    logger.info(f"Starting batch enrichment for {len(element_uuids)} elements in tenant {tenant_id}")

    # 1. Fetch all elements details
    elements_to_process = []
    # We can use fetch_objects with filter UUID contains_any
    try:
        collection = client.collections.get("CodeElement")
        # Chunking UUIDs for query if necessary (e.g. 100 at a time)
        chunk_size = 100
        fetched_objects = []
        for i in range(0, len(element_uuids), chunk_size):
            chunk = element_uuids[i:i+chunk_size]
            # Filters need weaviate.classes.query
            from weaviate.classes.query import Filter
            response = await asyncio.to_thread(
                collection.with_tenant(tenant_id).query.fetch_objects,
                filters=Filter.by_id().contains_any(chunk),
                limit=len(chunk),
                include_vector=True
            )
            fetched_objects.extend(response.objects)
    except Exception as e:
        logger.error(f"Error fetching elements for batch enrichment: {e}")
        return {"success_count": 0, "failed_count": len(element_uuids)}

    # 2. Process items
    objects_to_update = []

    async def process_single_item(obj):
        try:
            props = obj.properties
            uuid_str = str(obj.uuid)
            current_desc = props.get("llm_description", "")
            has_vector = obj.vector is not None

            if current_desc and current_desc != "[Description not generated]" and has_vector:
                return None # Already done

            element_type = props.get("element_type", "element")
            name = props.get("name", "Unknown")
            readable_id = props.get("readable_id", "N/A")
            signature = props.get("signature", "")
            docstring = props.get("docstring", "")
            code_snippet = props.get("code_snippet", "")

            content_to_embed = f"{element_type.capitalize()}: {name}\nReadable ID: {readable_id}\n"
            if signature: content_to_embed += f"Signature: {signature}\n"
            if docstring: content_to_embed += f"Docstring: {docstring}\n"
            content_to_embed += f"Code:\n{code_snippet}"

            prompt = f"Provide a concise, one-sentence description of the following Python {element_type} named '{name}'.\n"
            if readable_id != "N/A": prompt += f"Readable ID: {readable_id}\n"
            if signature: prompt += f"Signature: {signature}\n"
            prompt += f"Code:\n```python\n{code_snippet}\n```\n"
            if docstring: prompt += f"Docstring:\n{docstring}\n"
            prompt += "\nConcise description:"

            # Run API calls
            new_vector = None
            new_description = current_desc

            # Embed
            try:
                embedding_result = await asyncio.to_thread(
                    genai.embed_content,
                    model=embedding_model_name,
                    content=content_to_embed,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                new_vector = embedding_result.get("embedding")
            except Exception as e:
                logger.warning(f"Embedding failed for {uuid_str}: {e}")

            # Generate Description
            try:
                description_response = await asyncio.to_thread(
                    model.generate_content, prompt
                )
                new_description = description_response.text.strip()
            except Exception as e:
                logger.warning(f"Description generation failed for {uuid_str}: {e}")

            if new_vector or (new_description and new_description != current_desc):
                updated_props = props.copy()
                updated_props["llm_description"] = new_description
                return {
                    "uuid": uuid_str,
                    "properties": updated_props,
                    "vector": new_vector if new_vector else obj.vector
                }
            return None
        except Exception as e:
            logger.error(f"Error processing item {obj.uuid}: {e}")
            return None

    # Run processing in parallel
    # Limit concurrency
    sem = asyncio.Semaphore(10) # 10 concurrent LLM calls
    async def limited_process(obj):
        async with sem:
            return await process_single_item(obj)

    tasks = [limited_process(obj) for obj in fetched_objects]
    results = await asyncio.gather(*tasks)

    for res in results:
        if res:
            objects_to_update.append(res)

    # 3. Batch Update
    if objects_to_update:
        logger.info(f"Updating {len(objects_to_update)} elements in batch for tenant {tenant_id}")
        # Use add_objects_batch from weaviate_client
        # Note: add_objects_batch uses insert_many. If objects exist, we need to know if it upserts.
        # Weaviate `insert_many` typically fails on existing UUIDs unless we delete first?
        # Actually, `client.batch` (v3) allowed replace. v4 `insert_many` might not.
        # We might need to iterate and replace if batch update isn't straightforward or use `batch.add_object` loop.

        # Let's use `add_objects_batch` but verify implementation in `weaviate_client.py`.
        # `add_objects_batch` uses `collection.data.insert_many`.
        # Weaviate v4 documentation says insert_many is for creation.
        # For updates, we might have to use loop with replace, but we can parallelize the loop requests (HTTP/2 multiplexing).
        # OR use batch.

        # Re-implement batching here using client.batch (v4 dynamic batch) if possible for upsert?
        # The v4 client has `batch.add_object`. If UUID exists, it depends on config?
        # Usually it's safer to use replace() loop for updates if batch doesn't support upsert explicitly.
        # BUT, to solve Scenario 17, we want to reduce round trips.
        # Since we are already running inside `mcp_server`, let's just use `add_objects_batch` assuming we delete first?
        # No, deleting destroys references! We must UPDATE (replace).

        # So we cannot use `insert_many`.
        # We must use `replace`.
        # To optimize, we can run `replace` calls in parallel using asyncio.gather.

        update_tasks = []
        collection = client.collections.get("CodeElement")

        async def update_single(obj_data):
            try:
                await asyncio.to_thread(
                    collection.with_tenant(tenant_id).data.replace,
                    uuid=obj_data["uuid"],
                    properties=obj_data["properties"],
                    vector=obj_data["vector"]
                )
                return True
            except Exception as e:
                logger.error(f"Update failed for {obj_data['uuid']}: {e}")
                return False

        # Concurrency for DB writes
        db_sem = asyncio.Semaphore(20)
        async def limited_update(obj_data):
            async with db_sem:
                return await update_single(obj_data)

        for obj_data in objects_to_update:
            update_tasks.append(limited_update(obj_data))

        update_results = await asyncio.gather(*update_tasks)
        success_count = sum(1 for r in update_results if r)
        return {"success_count": success_count, "failed_count": len(update_tasks) - success_count}

    return {"success_count": 0, "failed_count": 0}


async def _scan_cleanup_and_upload(
    client, directory: str, tenant_id: str, specific_files: list[str] = None
) -> tuple[str, list[str]]:
    """
    Internal helper to perform scan, cleanup, and upload for a specific tenant.

    Args:
        client: Weaviate client.
        directory: Root directory of the codebase.
        tenant_id: Tenant ID (codebase name).
        specific_files: Optional list of absolute file paths to scan. If provided, only these files are processed.

    Returns status message and list of processed element UUIDs.
    """
    logger.info(
        f"Starting _scan_cleanup_and_upload for tenant '{tenant_id}' in directory '{directory}'"
    )
    if specific_files:
        logger.info(f"Targeted scan for {len(specific_files)} specific files.")

    processed_uuids_list = []
    final_status_messages = []

    try:
        logger.info(
            f"--- Scan Pass 1: File Discovery and Deletion Check for tenant '{tenant_id}' ---"
        )
        code_files_to_batch = []
        code_elements_to_batch = []
        references_to_batch = []
        processed_element_uuids = []
        files_skipped_count = 0
        files_processed_count = 0

        # Determine files to check
        local_python_files = []
        stored_files_data = {}
        files_to_delete_completely = []

        if specific_files:
            # Targeted update
            local_python_files = [f for f in specific_files if f.endswith(".py") and os.path.exists(f)]
            # We only need metadata for these files
            # Convert to relative paths for metadata lookup
            rel_paths = []
            for f in local_python_files:
                try:
                    rel_paths.append(os.path.relpath(f, start=directory).replace(os.sep, "/"))
                except ValueError:
                     # Path is not under directory, skip or handle?
                     logger.warning(f"File {f} is not under {directory}. Skipping.")
                     pass

            stored_files_data = get_code_files_metadata(client, tenant_id, rel_paths)

            # Check for deletions in specific list (if file doesn't exist anymore)
            # But specific_files usually come from existing events.
            # If deleted event, the file won't exist.
            # If the caller passes a deleted file in specific_files, we handle it?
            # Usually the watcher calls delete separately.
            # But if we want to be safe:
            for f in specific_files:
                if not os.path.exists(f):
                     try:
                        rel = os.path.relpath(f, start=directory).replace(os.sep, "/")
                        files_to_delete_completely.append(rel)
                     except ValueError:
                         pass

        else:
            # Full scan
            stored_files_data = get_all_code_files(client, tenant_id)
            local_python_files = await find_python_files(directory)
            local_file_paths = set()
            for f in local_python_files:
                 try:
                    rel = os.path.relpath(f, start=directory).replace(os.sep, "/")
                    local_file_paths.add(rel)
                 except ValueError:
                     pass

            # Identify deleted files (only relevant for full scan)
            for file_path in stored_files_data.keys():
                if file_path not in local_file_paths:
                    logger.info(f"File deleted locally: {file_path} (Tenant: {tenant_id})")
                    files_to_delete_completely.append(file_path)

        files_to_process = []
        files_to_delete_elements_for = []

        logger.debug("--- Starting file comparison loop ---")
        for abs_file_path in local_python_files:
            current_mtime_dt = None
            stored_mtime_dt = None
            comparison_result = "ERROR_before_comparison"

            try:
                relative_file_path_for_lookup = os.path.relpath(abs_file_path, start=directory).replace(os.sep, "/")
            except ValueError:
                continue

            try:
                current_mtime_float = os.path.getmtime(abs_file_path)
                current_mtime_dt = datetime.datetime.fromtimestamp(
                    current_mtime_float, tz=datetime.timezone.utc
                )
                stored_mtime_dt = stored_files_data.get(relative_file_path_for_lookup)

                if stored_mtime_dt is None:
                    comparison_result = "NEW"
                    files_to_process.append(
                        (relative_file_path_for_lookup, abs_file_path, current_mtime_dt, "new")
                    )
                elif current_mtime_dt > stored_mtime_dt:
                    comparison_result = "MODIFIED"
                    files_to_process.append(
                        (relative_file_path_for_lookup, abs_file_path, current_mtime_dt, "modified")
                    )
                    files_to_delete_elements_for.append(relative_file_path_for_lookup)
                else:
                    comparison_result = "UNCHANGED"
                    files_skipped_count += 1

                logger.debug(
                    f"--- DEBUG SCAN LOOP: '{relative_file_path_for_lookup}': {comparison_result}"
                )

            except FileNotFoundError:
                logger.warning(
                    f"File not found during mtime check: {abs_file_path}"
                )
                comparison_result = "ERROR_file_not_found"
            except Exception as e:
                logger.error(
                    f"Error checking mtime for {relative_file_path_for_lookup}: {e}"
                )
                comparison_result = f"ERROR_exception:_{e}"

        logger.debug("--- Finished file comparison loop ---")
        logger.debug(f"Files marked for processing: {len(files_to_process)}")

        files_to_clear = set(files_to_delete_elements_for + files_to_delete_completely)

        # Parallel Processing
        logger.info("--- Scan Pass 2: Code Parsing and Element Extraction (Parallel) ---")

        loop = asyncio.get_running_loop()

        # Helper to read file and prep for worker
        async def prep_and_run_worker(file_info):
            rel_path, abs_path, mtime, status = file_info
            content = await read_file_content(abs_path)
            if content is None:
                return None

            file_uuid = generate_uuid5(rel_path)

            # Add to batch for CodeFile object
            code_file_obj = {
                "uuid": file_uuid,
                "properties": {
                    "path": rel_path,
                    "last_modified": mtime.isoformat(),
                },
            }

            # Run CPU bound task in shared executor
            try:
                executor = get_shared_executor()
                elements, references = await loop.run_in_executor(
                    executor, process_file_worker, rel_path, content, file_uuid, mtime
                )
                return code_file_obj, elements, references
            except Exception as e:
                logger.error(f"Error processing file {rel_path}: {e}")
                return None

        # Create tasks
        tasks = [prep_and_run_worker(info) for info in files_to_process]

        # Gather results
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                c_file, c_elements, c_refs = res
                code_files_to_batch.append(c_file)
                code_elements_to_batch.extend(c_elements)
                references_to_batch.extend(c_refs)
                processed_element_uuids.extend([el["uuid"] for el in c_elements if "uuid" in el])
                files_processed_count += 1


        scan_status_message = f"Scan completed for tenant '{tenant_id}'. Files processed: {files_processed_count}. Files skipped: {files_skipped_count}."
        logger.info(scan_status_message)

        final_status_messages.append(scan_status_message)
        processed_uuids_list = processed_element_uuids

        logger.info(f"Performing deletions for tenant '{tenant_id}'...")
        delete_success = True

        files_needing_element_deletion = list(
            files_to_clear - set(files_to_delete_completely)
        )
        if files_needing_element_deletion:
            logger.info(
                f"Deleting elements for {len(files_needing_element_deletion)} modified files in tenant '{tenant_id}'..."
            )
            # Parallelize deletions? They are network bound.
            # But delete_elements_by_file_path is sync (wrapped in to_thread maybe?).
            # For now, let's keep it simple, but we can do simple loop.
            for file_path in files_needing_element_deletion:
                if not delete_elements_by_file_path(client, tenant_id, file_path):
                    logger.warning(
                        f"Failed to delete elements for modified file: {file_path} in tenant '{tenant_id}'"
                    )
                    delete_success = False

        if files_to_delete_completely:
            logger.info(
                f"Deleting {len(files_to_delete_completely)} completely removed files and their elements in tenant '{tenant_id}'..."
            )
            for file_path in files_to_delete_completely:
                if not delete_elements_by_file_path(client, tenant_id, file_path):
                    logger.warning(
                        f"Failed to delete elements for removed file: {file_path} in tenant '{tenant_id}'"
                    )
                    delete_success = False
                if not delete_code_file(client, tenant_id, file_path):
                    logger.warning(
                        f"Failed to delete CodeFile object for removed file: {file_path} in tenant '{tenant_id}'"
                    )
                    delete_success = False

        if not delete_success:
            final_status_messages.append("Warning: Some deletions failed.")
            return ", ".join(final_status_messages), processed_uuids_list

        logger.info(f"Performing batch uploads for tenant '{tenant_id}'...")
        upload_success = True

        if code_files_to_batch:
            logger.info(
                f"Uploading {len(code_files_to_batch)} CodeFile objects for tenant '{tenant_id}'..."
            )
            file_batch_result = add_objects_batch(
                client, code_files_to_batch, "CodeFile", tenant_id
            )
            if file_batch_result.get("status") != "success":
                logger.error(
                    f"CodeFile batch upload failed for tenant '{tenant_id}': {file_batch_result.get('message')}"
                )
                upload_success = False
            else:
                final_status_messages.append(
                    f"Uploaded {file_batch_result.get('count', 0)} CodeFiles."
                )

        if code_elements_to_batch:
            logger.info(
                f"Uploading {len(code_elements_to_batch)} CodeElement objects for tenant '{tenant_id}'..."
            )
            element_batch_result = add_objects_batch(
                client, code_elements_to_batch, "CodeElement", tenant_id
            )
            if element_batch_result.get("status") != "success":
                logger.error(
                    f"CodeElement batch upload failed for tenant '{tenant_id}': {element_batch_result.get('message')}"
                )
                upload_success = False
            else:
                final_status_messages.append(
                    f"Uploaded {element_batch_result.get('count', 0)} CodeElements."
                )

        if references_to_batch:
            logger.info(
                f"Uploading {len(references_to_batch)} references for tenant '{tenant_id}'..."
            )
            ref_batch_result = add_references_batch(
                client, references_to_batch, tenant_id
            )
            if ref_batch_result.get("status") != "success":
                logger.error(
                    f"Reference batch upload failed for tenant '{tenant_id}': {ref_batch_result.get('message')}"
                )
                upload_success = False
            else:
                final_status_messages.append(
                    f"Uploaded {ref_batch_result.get('count', 0)} references."
                )

        if not upload_success:
            final_status_messages.insert(0, "ERROR: One or more batch uploads failed.")
        else:
            final_status_messages.append("Uploads completed.")

        final_message = ", ".join(msg for msg in final_status_messages if msg)
        logger.info(
            f"_scan_cleanup_and_upload finished for tenant '{tenant_id}'. Status: {final_message}"
        )
        return final_message, processed_uuids_list

    except Exception as e:
        error_message = f"ERROR: Unexpected error in _scan_cleanup_and_upload for tenant '{tenant_id}': {type(e).__name__}: {e}"
        logger.exception(error_message)
        return error_message, []
