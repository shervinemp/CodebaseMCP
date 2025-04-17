import os
import ast
import google.generativeai as genai
import google.api_core.exceptions
import datetime
import asyncio
import re
from dotenv import load_dotenv
from weaviate_client import (  # Use relative import
    create_weaviate_client,
    get_all_code_files,
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


logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded for code_scanner.")


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
def find_python_files(directory):
    """Finds all python files recursively in the given directory"""

    assert os.path.isabs(directory), "Directory must be absolute."

    python_files = []
    start_dir = os.path.abspath(directory)
    python_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(start_dir)
        for f in files
        if f.endswith(".py")
    ]

    return python_files


def read_file_content(abs_path):
    """Reads the content of a file at the given absolute path."""
    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_code(file_content, file_path, verbose=VERBOSE_LOGGING):
    """Parses code using AST. file_path is used for error reporting."""
    try:
        return ast.parse(file_content, filename=file_path)
    except SyntaxError as e:
        if verbose:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
        return None


# --- AST Visitor (Structural Analysis Only) ---
class CodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path, code_content, file_uuid):
        """Initializes the visitor."""
        logger.debug(
            f"CodeVisitor initialized with file_path: {file_path}, file_uuid: {file_uuid}"
        )
        self.file_path = file_path
        self.code_content = code_content
        self.file_uuid = file_uuid
        self.elements = []
        self.references = []
        self.element_uuid_map = {}
        self.scope_stack = []
        self.current_attributes = []
        try:
            self.file_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.file_path), tz=datetime.timezone.utc
            )
        except FileNotFoundError:
            logger.error(
                f"CodeVisitor: File not found for mtime check: {self.file_path}"
            )
            self.file_mtime = datetime.datetime.fromtimestamp(
                0, tz=datetime.timezone.utc
            )
        logger.debug(f"CodeVisitor instance created for {file_path}")

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
        logger.debug(
            f"Visiting FunctionDef: {node.name} at line {node.lineno} in {self.file_path}"
        )
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
        logger.debug(
            f"Visiting ClassDef: {node.name} at line {node.lineno} in {self.file_path}"
        )
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
    """
    Fetches an element from a specific tenant, generates its initial LLM description and embedding,
    and updates it in Weaviate. Runs LLM calls in a thread.
    Returns True on success/no action needed, False on failure.
    """
    if not GENERATE_LLM_DESCRIPTIONS or not model or not embedding_model_name:
        logger.debug(
            f"LLM generation disabled, skipping enrichment for {element_uuid} in tenant {tenant_id}"
        )
        return True

    logger.debug(f"Attempting enrichment for {element_uuid} in tenant {tenant_id}")
    try:
        element = get_element_details(client, tenant_id, element_uuid)
        if not element:
            logger.error(
                f"enrich_element: Element {element_uuid} not found in tenant {tenant_id}."
            )
            return False

        props = element.properties
        current_desc = props.get("llm_description", "")
        has_vector = element.vector is not None

        if (
            current_desc
            and current_desc != "[Description not generated]"
            and has_vector
        ):
            logger.debug(
                f"Element {element_uuid} in tenant {tenant_id} already enriched, skipping."
            )
            return True

        element_type = props.get("element_type", "element")
        name = props.get("name", "Unknown")
        readable_id = props.get("readable_id", "N/A")
        signature = props.get("signature", "")
        docstring = props.get("docstring", "")
        code_snippet = props.get("code_snippet", "")

        content_to_embed = (
            f"{element_type.capitalize()}: {name}\nReadable ID: {readable_id}\n"
        )
        if signature:
            content_to_embed += f"Signature: {signature}\n"
        if docstring:
            content_to_embed += f"Docstring: {docstring}\n"
        content_to_embed += f"Code:\n{code_snippet}"

        prompt = f"Provide a concise, one-sentence description of the following Python {element_type} named '{name}'.\n"
        if readable_id != "N/A":
            prompt += f"Readable ID: {readable_id}\n"
        if signature:
            prompt += f"Signature: {signature}\n"
        prompt += f"Code:\n```python\n{code_snippet}\n```\n"
        if docstring:
            prompt += f"Docstring:\n{docstring}\n"
        prompt += "\nConcise description:"

        new_vector = None
        new_description = current_desc

        try:
            logger.debug(
                f"Generating embedding for {element_uuid} ({name}) in tenant {tenant_id}"
            )
            embedding_result = await asyncio.to_thread(
                genai.embed_content,
                model=embedding_model_name,
                content=content_to_embed,
                task_type="RETRIEVAL_DOCUMENT",
            )
            new_vector = embedding_result.get("embedding")
        except google.api_core.exceptions.GoogleAPIError as embed_e:
            logger.warning(
                f"Embedding generation failed for {element_uuid} in tenant {tenant_id}: {type(embed_e).__name__}: {embed_e}"
            )
            raise

        try:
            logger.debug(
                f"Generating initial description for {element_uuid} ({name}) in tenant {tenant_id}"
            )
            # Wrap synchronous LLM call
            description_response = await asyncio.to_thread(
                model.generate_content, prompt
            )
            generated_desc = description_response.text.strip()
            new_description = generated_desc
            logger.debug(
                f"Generated description for {element_uuid} in tenant {tenant_id}: {new_description}"
            )
        except google.api_core.exceptions.GoogleAPIError as desc_e:
            logger.warning(
                f"Initial description generation failed for {element_uuid} in tenant {tenant_id}: {type(desc_e).__name__}: {desc_e}"
            )
            # Don't raise, just proceed without description if it fails
            new_description = current_desc  # Keep old description if generation fails

        if new_vector or (new_description and new_description != current_desc):
            props_to_update = props.copy()
            props_to_update["llm_description"] = new_description
            logger.debug(
                f"Updating element {element_uuid} in tenant {tenant_id} with new description/vector."
            )
            try:
                # Wrap synchronous Weaviate call
                collection = client.collections.get("CodeElement")
                await asyncio.to_thread(
                    collection.with_tenant(
                        tenant_id
                    ).data.replace,  # Corrected: replace was already wrapped, no change needed here. Retaining original wrap.
                    uuid=element_uuid,
                    properties=props_to_update,
                    vector=(new_vector if new_vector else element.vector),
                )
                return True
            except Exception as update_e:
                logger.error(
                    f"Failed to update element {element_uuid} in tenant {tenant_id} after enrichment: {type(update_e).__name__}: {update_e}"
                )
                return False
        else:
            logger.debug(
                f"No changes needed for {element_uuid} in tenant {tenant_id} after enrichment attempt."
            )
            return True

    except Exception as e:
        logger.exception(
            f"Unexpected error during description refinement for {element_uuid} in tenant {tenant_id}: {type(e).__name__}: {e}"
        )
        return False


async def _scan_cleanup_and_upload(
    client, directory: str, tenant_id: str
) -> tuple[str, list[str]]:
    """
    Internal helper to perform scan, cleanup, and upload for a specific tenant.
    Returns status message and list of processed element UUIDs.
    """
    logger.info(
        f"Starting _scan_cleanup_and_upload for tenant '{tenant_id}' in directory '{directory}'"
    )
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

        stored_files_data = get_all_code_files(client, tenant_id)
        local_python_files = find_python_files(directory)
        local_file_paths = set(local_python_files)
        files_to_process = []
        files_to_delete_elements_for = []
        files_to_delete_completely = []
        debug_info_for_status = ""

        for file_path in stored_files_data.keys():
            if file_path not in local_file_paths:
                logger.info(f"File deleted locally: {file_path} (Tenant: {tenant_id})")
                files_to_delete_completely.append(file_path)

        logger.debug("--- Starting file comparison loop ---")
        for file_path in local_python_files:
            current_mtime_dt = None
            stored_mtime_dt = None
            comparison_result = "ERROR_before_comparison"
            project_root = "c:/Users/sherv/Desktop/Projects/ragcode"
            abs_file_path_for_mtime = os.path.abspath(
                os.path.join(project_root, file_path)
            )
            relative_file_path_for_lookup = file_path

            try:
                current_mtime_float = os.path.getmtime(abs_file_path_for_mtime)
                current_mtime_dt = datetime.datetime.fromtimestamp(
                    current_mtime_float, tz=datetime.timezone.utc
                )
                stored_mtime_dt = stored_files_data.get(relative_file_path_for_lookup)

                if stored_mtime_dt is None:
                    comparison_result = "NEW"
                    files_to_process.append(
                        (relative_file_path_for_lookup, current_mtime_dt, "new")
                    )
                elif current_mtime_dt > stored_mtime_dt:
                    comparison_result = "MODIFIED"
                    files_to_process.append(
                        (relative_file_path_for_lookup, current_mtime_dt, "modified")
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
                    f"File not found during mtime check: {abs_file_path_for_mtime}"
                )
                comparison_result = "ERROR_file_not_found"
            except Exception as e:
                logger.error(
                    f"Error checking mtime for {relative_file_path_for_lookup}: {e}"
                )
                comparison_result = f"ERROR_exception:_{e}"

        logger.debug("--- Finished file comparison loop ---")
        logger.debug(f"Files marked for processing: {files_to_process}")

        files_to_clear = set(files_to_delete_elements_for + files_to_delete_completely)

        logger.info("--- Scan Pass 2: Code Parsing and Element Extraction ---")
        for file_path_rel, mtime_dt, status in files_to_process:
            file_content = read_file_content(file_path_rel)
            if file_content is None:
                continue

            file_uuid = generate_uuid5(file_path_rel)
            code_files_to_batch.append(
                {
                    "uuid": file_uuid,
                    "properties": {
                        "path": file_path_rel,
                        "last_modified": mtime_dt.isoformat(),
                    },
                }
            )

            parsed_code = parse_code(file_content, file_path_rel, VERBOSE_LOGGING)
            if parsed_code is None:
                continue

            visitor = CodeVisitor(file_path_rel, file_content, file_uuid)
            try:
                visitor.visit(parsed_code)
            except Exception as visit_e:
                logger.exception(f"Error visiting nodes in {file_path_rel}: {visit_e}")
                continue

            code_elements_to_batch.extend(visitor.elements)
            processed_element_uuids.extend(
                [el["uuid"] for el in visitor.elements if "uuid" in el]
            )
            references_to_batch.extend(visitor.references)
            files_processed_count += 1

        scan_status_message = f"Scan completed for tenant '{tenant_id}'. Files processed: {files_processed_count}. Files skipped: {files_skipped_count}."
        scan_status_message += debug_info_for_status
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
