import os
import logging
import re
import ast
from typing import Any, Dict, Optional, List  # Added List

# Assuming WeaviateManager might be needed if _process_element_properties evolves
# from .weaviate_client import WeaviateManager

logger = logging.getLogger(__name__)

# --- Helper Functions for Output Processing / Analysis ---


def _shorten_file_path(
    file_path: str | None, codebase_root_dir: str | None
) -> str | None:
    """Converts an absolute path to a path relative to the codebase root directory."""
    if not file_path or not codebase_root_dir:
        return file_path
    try:
        # Ensure both paths are absolute and normalized
        abs_path = os.path.abspath(os.path.normpath(file_path))
        codebase_root_norm = os.path.abspath(os.path.normpath(codebase_root_dir))

        # Check if the path is within the root directory
        if os.path.commonpath([abs_path, codebase_root_norm]) == codebase_root_norm:
            relative_path = os.path.relpath(abs_path, start=codebase_root_norm)
            # Convert to forward slashes for consistency
            final_path = relative_path.replace(os.sep, "/")
            return (
                final_path if final_path != "." else "./"
            )  # Handle case where path is the root itself
        else:
            # Path is outside the root, return normalized absolute path with forward slashes
            logger.debug(
                f"Path {abs_path} is outside codebase root {codebase_root_norm}. Returning absolute path."
            )
            return abs_path.replace(os.sep, "/")
    except ValueError:  # Can happen if paths are on different drives on Windows
        logger.warning(
            f"Could not determine relative path for {file_path} against root {codebase_root_dir}. Returning original."
        )
        return file_path
    except Exception as e:
        logger.error(
            f"Error shortening path {file_path} with root {codebase_root_dir}: {e}"
        )
        return file_path  # Return original path on error


def _trim_string(value: Any) -> Any:
    """Trims leading/trailing whitespace if the value is a string."""
    if isinstance(value, str):
        return value.strip()
    return value


async def _process_element_properties(
    # manager: WeaviateManager, # Manager not directly used here, only its data via caller
    tenant_id: str,  # Keep tenant_id for logging context if needed
    properties: dict[str, Any],
    uuid: str,
    view_type: str = "list",
    codebase_root_dir: str | None = None,
) -> dict[str, Any]:
    """Cleans, filters, and enhances properties for API output, tailored by view_type."""
    # Note: This function is async only because it might potentially call async helpers in the future.
    # Currently, it only calls sync helpers (_shorten_file_path, _trim_string).
    logger.debug(
        f"_process_element_properties START for tenant {tenant_id}, uuid {uuid}, view_type {view_type}, root={codebase_root_dir}"
    )
    logger.debug(f"  Input properties: {properties}")

    if not properties:
        logger.debug("_process_element_properties END (empty input)")
        return {}

    processed = {}

    # Define fields for different views
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
    processed["uuid"] = uuid  # Always include UUID
    target_fields = detail_view_fields if view_type == "detail" else list_view_fields

    for key, value in properties.items():
        output_key = key
        # Map internal names to user-facing names if needed
        if key == "element_type" and "type" in target_fields:
            output_key = "type"
        if key == "file_path" and "file" in target_fields:
            output_key = "file"

        # Skip fields not relevant for the current view
        if output_key not in target_fields:
            continue

        processed_value = value
        # Process specific fields
        if output_key == "file":
            # This helper is sync
            processed_value = _shorten_file_path(value, codebase_root_dir)
        elif key in [
            "user_clarification",
            "code_snippet",
            "llm_description",
            "docstring",
        ]:
            # This helper is sync
            processed_value = _trim_string(value)

        # Special handling for 'description' in list view
        if output_key == "description" and view_type == "list":
            llm_desc = _trim_string(properties.get("llm_description", ""))
            docstr = _trim_string(properties.get("docstring", ""))
            processed_value = (
                llm_desc or docstr or None
            )  # Prioritize LLM desc, then docstring
        elif key in ["llm_description", "docstring"] and view_type == "list":
            continue  # Don't include raw llm_desc/docstring in list view if description is handled

        # Skip empty optional fields in list view (except core fields)
        if (
            view_type == "list"
            and not processed_value
            and output_key not in {"name", "type", "file", "uuid"}
        ):
            continue

        processed[output_key] = processed_value

    # Ensure list view always has a description field, even if None
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
        # Fallback to regex on syntax error
        return re.findall(r"\b[a-zA-Z_]\w*\b", code_snippet)
    except Exception as e:
        logger.error(f"Error extracting identifiers using AST: {e}")
        # Fallback to regex on other errors
        return re.findall(r"\b[a-zA-Z_]\w*\b", code_snippet)

    # Basic filtering of common Python keywords
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
        "async",
        "await",
        "and",
        "or",
        "not",
        "in",
        "is",
        "del",
        "global",
        "nonlocal",
        "assert",
        "break",
        "continue",
        "raise",
    }
    filtered_identifiers = list(identifiers - keywords)
    logger.debug(f"Extracted identifiers using AST: {filtered_identifiers}")
    return filtered_identifiers
