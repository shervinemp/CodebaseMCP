import os
import ast
import pytest
import shutil
from weaviate.util import generate_uuid5
from weaviate.classes.tenants import Tenant  # Added import

# Update imports to reflect new structure
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.code_analysis_mcp.code_scanner import (
    find_python_files,
    read_file_content,
    parse_code,
    # scan_directory, # Function removed, logic inlined in _scan_cleanup_and_upload
    # enrich_element # Tested in test_rag.py potentially
)
from src.code_analysis_mcp.weaviate_client import (
    create_weaviate_client,
    find_element_by_name,
    delete_elements_by_file_path,
    delete_code_file,
    create_schema,
    add_objects_batch,  # Needed for test setup
    add_references_batch,  # Needed for test setup
)


# --- Test Setup ---
TEST_DIR = "pytest_temp_files_analysis"  # Use different dir to avoid conflicts
TEST_FILE_SIMPLE = os.path.join(TEST_DIR, "simple_module.py")
TEST_FILE_CLASS = os.path.join(TEST_DIR, "class_module.py")
TEST_FILE_SYNTAX_ERROR = os.path.join(TEST_DIR, "syntax_error_module.py")
TEST_FILE_EMPTY = os.path.join(TEST_DIR, "empty_module.py")


@pytest.fixture(scope="module", autouse=True)
def setup_test_files_and_cleanup():
    """Create temporary directory and test files for the module."""
    print(f"\nSetting up test directory: {TEST_DIR}")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    # simple_module.py content
    with open(TEST_FILE_SIMPLE, "w", encoding="utf-8") as f:
        f.write(
            "@my_decorator\n"
            "def simple_function(x: int) -> int:\n"
            '    """Simple docstring."""\n'
            "    y = x + 1\n"
            "    print(y)\n"
            "    obj = SomeClass()\n"
            "    res = obj.calculate(y)\n"
            "    return res\n"
        )

    # class_module.py content
    with open(TEST_FILE_CLASS, "w", encoding="utf-8") as f:
        f.write(
            "import os\n\n"
            "@class_decorator\n"
            "class MyClass(object, AnotherBase):\n"  # Added multiple base classes
            '    """Class docstring."""\n'
            "    class_attr = 10\n"
            "    def method(self, val):\n"
            "        print(self.class_attr)\n"
            '        return os.path.join("test", val)\n'
        )

    # syntax_error_module.py content
    with open(TEST_FILE_SYNTAX_ERROR, "w", encoding="utf-8") as f:
        f.write("def bad_syntax(\n    pass\n")

    # empty_module.py content
    with open(TEST_FILE_EMPTY, "w", encoding="utf-8") as f:
        f.write("")

    yield

    print(f"\nCleaning up test directory: {TEST_DIR}")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


@pytest.fixture(scope="module")
def test_client():
    """Provides a connected Weaviate client for the test module."""
    client = create_weaviate_client()
    if not client:
        pytest.fail("Failed to create Weaviate client.")
    client.connect()
    if not client.is_connected():
        pytest.fail("Failed to connect to Weaviate.")

    print("\nDeleting existing schema (if any)...")
    client.collections.delete_all()
    print("Creating Weaviate schema for tests...")
    create_schema(client)

    yield client

    print("\nClosing Weaviate client connection...")
    client.close()


# --- Basic Function Tests ---


def test_find_python_files():
    files = find_python_files(TEST_DIR)
    expected_files = {
        os.path.normpath(p).replace(os.sep, "/")  # Normalize and use forward slash
        for p in [
            TEST_FILE_SIMPLE,
            TEST_FILE_CLASS,
            TEST_FILE_SYNTAX_ERROR,
            TEST_FILE_EMPTY,
        ]
    }
    actual_files = {
        p.replace(os.sep, "/") for p in files
    }  # Ensure actual uses forward slash
    assert actual_files == expected_files


def test_read_file_content():
    content = read_file_content(TEST_FILE_SIMPLE)
    assert "def simple_function(x: int) -> int:" in content
    assert '"""Simple docstring."""' in content


def test_parse_code_success():
    content = read_file_content(TEST_FILE_SIMPLE)
    tree = parse_code(content, TEST_FILE_SIMPLE)
    assert isinstance(tree, ast.AST)


def test_parse_code_syntax_error():
    content = read_file_content(TEST_FILE_SYNTAX_ERROR)
    tree = parse_code(content, TEST_FILE_SYNTAX_ERROR)
    assert tree is None


def test_parse_code_empty_file():
    content = read_file_content(TEST_FILE_EMPTY)
    tree = parse_code(content, TEST_FILE_EMPTY)
    assert isinstance(tree, ast.AST)
    assert len(tree.body) == 0


# --- Integration Test for Scan + Upload --- # Removed test that used scan_directory
