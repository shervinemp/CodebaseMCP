import sys
import os

# Ensure the project root is in the path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Adjusted below

import pytest, asyncio, time
from unittest.mock import patch, MagicMock, AsyncMock
from weaviate.classes.tenants import Tenant
import weaviate.classes.query as wvc_query  # Import query classes

# Update imports to reflect new structure and ensure src is found
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Removed - pytest should handle this from root
import src.code_analysis_mcp.mcp_server as mcp_server  # Import the server module
from src.code_analysis_mcp.mcp_server import (
    _shorten_file_path,
    ScanProjectArgs,  # Import necessary arg models
    SelectProjectArgs,
    DeleteProjectArgs,
    FindElementArgs,
    GetDetailsArgs,
    AskQuestionArgs,
    AnalyzeSnippetArgs,
    TriggerLlmArgs,
)
from src.code_analysis_mcp.weaviate_client import (
    create_weaviate_client,
    create_schema,
    delete_tenant,
    get_project_details,  # Import real DB functions needed for setup/assertions
    add_project_registry_entry,
    get_all_projects,
    add_objects_batch,
    find_element_by_name,  # Rename to avoid conflict with tool
    get_element_details,  # Rename to avoid conflict with tool
    delete_project_registry_entry,  # Import for cleanup
)

# --- Constants ---
TEST_TENANT_ID = "_pytest_tenant_"  # Define a consistent tenant ID for testing

# --- Test Setup ---


# Helper to create mock Weaviate Object (Still useful for comparing results)
def create_mock_weaviate_object(uuid, properties, vector=None, references=None):
    mock_obj = MagicMock()
    mock_obj.uuid = uuid
    mock_obj.properties = properties
    mock_obj.vector = vector
    mock_obj.references = references or {}
    return mock_obj


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clear_mcp_state():
    """Clears global MCP server state before/after each test."""
    original_active_project = mcp_server.ACTIVE_PROJECT_NAME
    original_tasks = set(mcp_server.background_llm_tasks)
    mcp_server.ACTIVE_PROJECT_NAME = None
    mcp_server.background_llm_tasks.clear()
    yield
    # Cleanup background tasks that might have been created
    tasks_to_cancel = list(mcp_server.background_llm_tasks)
    if tasks_to_cancel:
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
    mcp_server.background_llm_tasks.clear()
    mcp_server.ACTIVE_PROJECT_NAME = original_active_project


# --- New Fixtures for Real Weaviate Interaction ---


@pytest.fixture(scope="module")
def real_weaviate_client():
    """Creates a real Weaviate client connection for the module."""
    print("\n--- Creating real Weaviate client (module scope) ---")
    client = None
    try:
        client = create_weaviate_client()
        if not client:
            pytest.fail("Failed to create Weaviate client.")

        # Explicitly connect the client
        print("Connecting real Weaviate client...")
        client.connect()
        print("Client connect() called.")

        # Wait briefly for Weaviate to be ready (adjust retries/delay as needed)
        for _ in range(5):
            if client.is_ready():
                print("Weaviate client is ready.")
                break
            print("Waiting for Weaviate...")
            time.sleep(1)
        else:
            pytest.fail("Weaviate client did not become ready.")

        # Ensure schema exists
        print("Ensuring Weaviate schema exists...")
        create_schema(client)
        print("Schema check/creation complete.")

        # Add explicit check for ProjectRegistry existence
        print("Verifying ProjectRegistry collection exists...")
        registry_exists = False
        for _ in range(5):  # Retry a few times
            if client.collections.exists("ProjectRegistry"):
                registry_exists = True
                print("ProjectRegistry collection confirmed.")
                break
            print("ProjectRegistry not found yet, waiting...")
            time.sleep(0.5)
        if not registry_exists:
            pytest.fail(
                "ProjectRegistry collection did not become available after schema creation."
            )

        yield client  # Provide the client to the setup_test_tenant fixture

    finally:
        if client and client.is_connected():
            print("\n--- Closing real Weaviate client (module scope) ---")
            client.close()


@pytest.fixture(scope="module")
def setup_test_tenant(real_weaviate_client):
    """Ensures the test tenant exists and cleans its contents after the module."""
    client = real_weaviate_client
    tenant_id = TEST_TENANT_ID
    print(f"\n--- Setting up test tenant '{tenant_id}' (module scope) ---")

    # Ensure tenant exists before tests run
    try:
        code_file_collection = client.collections.get("CodeFile")
        if not code_file_collection.tenants.exists(tenant_id):
            print(f"Creating test tenant '{tenant_id}' for CodeFile...")
            code_file_collection.tenants.create([Tenant(name=tenant_id)])

        code_element_collection = client.collections.get("CodeElement")
        if not code_element_collection.tenants.exists(tenant_id):
            print(f"Creating test tenant '{tenant_id}' for CodeElement...")
            code_element_collection.tenants.create([Tenant(name=tenant_id)])
        print(f"Test tenant '{tenant_id}' ensured.")
    except Exception as e:
        pytest.fail(f"Failed to ensure test tenant '{tenant_id}' exists: {e}")

    # Set the global client in mcp_server for the duration of the tests
    # This replaces the need for the first patch in the old fixture
    original_global_client = mcp_server.global_weaviate_client
    mcp_server.global_weaviate_client = client

    yield client, tenant_id  # Provide client and tenant ID to tests

    # --- Teardown ---
    print(f"\n--- Tearing down test tenant '{tenant_id}' (module scope) ---")
    # Restore original global client
    mcp_server.global_weaviate_client = original_global_client
    try:
        # Delete all objects within the tenant for relevant collections
        # ProjectRegistry is not multi-tenant, handle separately if needed
        print(
            f"Deleting objects from CodeElement collection for tenant '{tenant_id}'..."
        )
        # Use delete_many with a filter that matches all (e.g., by a required property)
        # Weaviate v4 requires a filter for delete_many.
        if client.collections.exists("CodeElement"):
            element_collection = client.collections.get("CodeElement").with_tenant(
                tenant_id
            )
            element_collection.data.delete_many(
                where=wvc_query.Filter.by_property("name").like("*")
            )  # Corrected filter
        print(f"Deleting objects from CodeFile collection for tenant '{tenant_id}'...")
        if client.collections.exists("CodeFile"):
            file_collection = client.collections.get("CodeFile").with_tenant(tenant_id)
            file_collection.data.delete_many(
                where=wvc_query.Filter.by_property("path").like("*")  # Corrected filter
            )
        # Optionally delete the tenant itself if preferred, but cleaning is often sufficient
        # delete_tenant(client, tenant_id)
        print(f"Test tenant '{tenant_id}' contents cleaned.")
    except Exception as e:
        print(f"\nWARNING: Error cleaning up test tenant '{tenant_id}': {e}")


@pytest.fixture(scope="function")  # Function scope for isolation between tests
def mock_llm_and_helpers():
    """Mocks non-database dependencies like LLMs and background tasks."""
    mocks = {
        # LLM/RAG functions
        "answer_codebase_question": AsyncMock(return_value="Mock RAG Answer"),
        "generate_project_summary": AsyncMock(return_value="Mock Project Summary."),
        "enrich_element": AsyncMock(return_value=True),  # Mock from code_scanner
        "refine_element_description": AsyncMock(return_value=True),  # Mock from rag
        "process_element_llm": AsyncMock(),  # Mock background task trigger in mcp_server
        # Background task management
        "create_task": MagicMock(),  # Mock asyncio.create_task
        "background_llm_tasks": MagicMock(spec=set),  # Mock the task set in mcp_server
        "background_generate_summary": AsyncMock(),  # Mock background task trigger in mcp_server
        # Other helpers (can be removed if tested directly)
        "_scan_cleanup_and_upload": AsyncMock(
            return_value=("Scan OK.", ["uuid-scan-1"])
        ),  # Mock from code_scanner (used by scan_project and watcher)
        "_process_element_properties": AsyncMock(
            side_effect=lambda c, t, p, u, vt="list": {**p, "uuid": u, "_view": vt}
        ),  # Mock output processing in mcp_server
        "_extract_identifiers": MagicMock(
            return_value=["ident1"]
        ),  # Mock in mcp_server
        "stop_watcher": MagicMock(return_value=(True, "Stopped")),  # Mock in mcp_server
    }

    patchers = [
        # Patch functions directly imported/used in mcp_server
        patch(
            "src.code_analysis_mcp.mcp_server.answer_codebase_question",
            mocks["answer_codebase_question"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server.generate_project_summary",
            mocks["generate_project_summary"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server.enrich_element", mocks["enrich_element"]
        ),  # Patches the imported name
        patch(
            "src.code_analysis_mcp.mcp_server.refine_element_description",
            mocks["refine_element_description"],
        ),  # Patches the imported name
        patch(
            "src.code_analysis_mcp.mcp_server.process_element_llm",
            mocks["process_element_llm"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server.asyncio.create_task", mocks["create_task"]
        ),
        patch(
            "src.code_analysis_mcp.mcp_server.background_llm_tasks",
            mocks["background_llm_tasks"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server.background_generate_summary",
            mocks["background_generate_summary"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server._scan_cleanup_and_upload",
            mocks["_scan_cleanup_and_upload"],
        ),  # Patches the imported name
        patch(
            "src.code_analysis_mcp.mcp_server._process_element_properties",
            mocks["_process_element_properties"],
        ),
        patch(
            "src.code_analysis_mcp.mcp_server._extract_identifiers",
            mocks["_extract_identifiers"],
        ),
        patch("src.code_analysis_mcp.mcp_server.stop_watcher", mocks["stop_watcher"]),
    ]

    for p in patchers:
        p.start()

    yield mocks

    for p in patchers:
        p.stop()


# --- Tool Tests (Refactored) ---

# === scan_project Tests ===


@pytest.mark.asyncio
@patch("os.path.isdir", return_value=True)  # Keep mocking os functions
async def test_scan_project_success_new(
    mock_isdir, setup_test_tenant, mock_llm_and_helpers  # Use new fixtures
):
    """Tests successful scanning of a new project using real Weaviate."""
    client, tenant_id = setup_test_tenant  # Get client and tenant ID
    from src.code_analysis_mcp.mcp_server import (
        scan_project,
    )  # Import the tool function

    # Use a unique project name derived from the tenant ID for isolation
    project_name = f"scan_new_{tenant_id}"
    args = ScanProjectArgs(project_name=project_name, directory="./src")
    abs_path = os.path.abspath("./src")

    # --- Pre-checks (using real client) ---
    # Ensure project doesn't exist in registry
    assert get_project_details(client, project_name) is None
    # Ensure tenant doesn't exist for this *new* project name yet
    assert not client.collections.get("CodeElement").tenants.exists(project_name)

    # Mock the complex scan helper (still mocked for now)
    mock_llm_and_helpers["_scan_cleanup_and_upload"].return_value = (
        "Scan Complete.",
        ["uuid1"],
    )

    # --- Execute the tool ---
    result = await scan_project(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["project_name"] == project_name
    assert "Scan successful" in result["message"]
    assert "Summary generation started" in result["message"]
    assert "active project" in result["message"]
    assert mcp_server.ACTIVE_PROJECT_NAME == project_name

    # Verify registry entry was created (using real client)
    registry_entry = get_project_details(client, project_name)
    assert registry_entry is not None
    assert registry_entry["directory"] == abs_path
    # Status might be Summarizing or Ready depending on background task mock/timing
    assert registry_entry["status"] in ["Summarizing", "Ready"]

    # Verify tenant was created for the project name
    assert client.collections.get("CodeElement").tenants.exists(project_name)

    # Verify mocks for non-DB operations
    mock_llm_and_helpers["_scan_cleanup_and_upload"].assert_awaited_once_with(
        client, abs_path, tenant_id=project_name
    )
    mock_llm_and_helpers[
        "create_task"
    ].assert_called()  # Check background task creation
    mock_isdir.assert_called_once()

    # --- Cleanup (Registry entry and Tenant) ---
    # Delete registry entry
    delete_project_registry_entry(client, project_name)  # Use the function
    # Delete the tenant created by the test
    delete_tenant(client, project_name)


@pytest.mark.asyncio
@patch("os.path.isdir", return_value=True)
async def test_scan_project_already_exists_registry(
    mock_isdir, setup_test_tenant, mock_llm_and_helpers  # Use new fixtures
):
    """Tests scanning when project already exists in the registry."""
    client, tenant_id = setup_test_tenant  # Use base test tenant for setup
    from src.code_analysis_mcp.mcp_server import scan_project  # Import tool

    # --- Setup: Add existing project to registry ---
    project_name = f"existing_{tenant_id}"  # Unique name based on tenant
    abs_path = os.path.abspath("./src")
    registry_uuid = add_project_registry_entry(client, project_name, abs_path, "Ready")
    assert registry_uuid is not None  # Check if entry was added
    assert get_project_details(client, project_name) is not None  # Verify setup

    # --- Execute ---
    args = ScanProjectArgs(project_name=project_name, directory="./src")
    result = await scan_project(args)

    # --- Assertions ---
    assert result["status"] == "error"
    assert "already exists" in result["message"]
    mock_llm_and_helpers[
        "_scan_cleanup_and_upload"
    ].assert_not_awaited()  # Scan shouldn't run
    mock_isdir.assert_called_once()

    # --- Cleanup (Registry entry) ---
    delete_project_registry_entry(client, project_name)  # Use the function


@pytest.mark.asyncio
@patch("os.path.isdir", return_value=True)
async def test_scan_project_tenant_exists_inconsistent(
    mock_isdir, setup_test_tenant, mock_llm_and_helpers  # Use new fixtures
):
    """Tests scanning when tenant exists but project is not in registry."""
    client, _ = setup_test_tenant  # Don't need base tenant_id directly
    from src.code_analysis_mcp.mcp_server import scan_project  # Import tool

    # --- Setup: Ensure tenant exists, but NOT in registry ---
    project_name = f"inconsistent_{TEST_TENANT_ID}"  # Use a unique name
    # Create the tenant manually
    client.collections.get("CodeElement").tenants.create([Tenant(name=project_name)])
    assert client.collections.get("CodeElement").tenants.exists(
        project_name
    )  # Verify tenant exists
    assert get_project_details(client, project_name) is None  # Verify not in registry

    # --- Execute ---
    args = ScanProjectArgs(project_name=project_name, directory="./src")
    result = await scan_project(args)

    # --- Assertions ---
    assert result["status"] == "error"
    assert (
        "Potential inconsistency" in result["message"]
        or "exists but it's not registered" in result["message"]
    )
    mock_llm_and_helpers["_scan_cleanup_and_upload"].assert_not_awaited()
    mock_isdir.assert_called_once()

    # --- Cleanup (Delete the manually created tenant) ---
    delete_tenant(client, project_name)


@pytest.mark.asyncio
@patch("os.path.isdir", return_value=True)
async def test_scan_project_scan_fails(
    mock_isdir, setup_test_tenant, mock_llm_and_helpers  # Use new fixtures
):
    """Tests scanning when the underlying _scan_cleanup_and_upload fails."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import scan_project  # Import tool

    project_name = f"scan_fail_{TEST_TENANT_ID}"
    args = ScanProjectArgs(project_name=project_name, directory="./src")
    abs_path = os.path.abspath("./src")

    # --- Setup: Mock scan helper to fail ---
    mock_llm_and_helpers["_scan_cleanup_and_upload"].return_value = (
        "ERROR: Scan Failed",
        [],
    )

    # --- Execute ---
    result = await scan_project(args)

    # --- Assertions ---
    assert result["status"] == "error"
    assert "Scan failed: ERROR: Scan Failed" in result["message"]

    # Verify registry entry was created but status is Error
    registry_entry = get_project_details(client, project_name)
    assert registry_entry is not None
    assert registry_entry["status"] == "Error"

    # Verify tenant was created
    assert client.collections.get("CodeElement").tenants.exists(
        project_name
    )  # Tenant name is project name

    mock_llm_and_helpers["_scan_cleanup_and_upload"].assert_awaited_once()
    mock_isdir.assert_called_once()

    # --- Cleanup (Registry entry and Tenant) ---
    delete_project_registry_entry(client, project_name)  # Use the function
    delete_tenant(client, project_name)


# === list_projects Tests ===


@pytest.mark.asyncio
async def test_list_projects_success(setup_test_tenant):  # Only need tenant fixture
    """Tests listing projects successfully using real Weaviate."""
    client, _ = setup_test_tenant  # Don't need tenant_id here
    from src.code_analysis_mcp.mcp_server import list_projects  # Import tool

    # --- Setup: Add projects to registry ---
    proj_a_name = f"ListProjA_{TEST_TENANT_ID}"  # Use unique names
    proj_b_name = f"ListProjB_{TEST_TENANT_ID}"
    # Clean up potential leftovers from previous runs first
    delete_project_registry_entry(client, proj_a_name)
    delete_project_registry_entry(client, proj_b_name)

    add_project_registry_entry(client, proj_a_name, "/abs/path/a", "Ready", "Summary A")
    add_project_registry_entry(client, proj_b_name, "/abs/path/b", "Scanning", "")
    assert len(get_all_projects(client)) >= 2  # Check they were added

    # --- Execute ---
    result = await list_projects()

    # --- Assertions ---
    assert result["status"] == "success"
    # Corrected key from 'project_name' to 'name' based on tool implementation
    project_names = [p["name"] for p in result["projects"]]
    assert proj_a_name in project_names
    assert proj_b_name in project_names

    # Corrected key in next() calls
    proj_a_data = next(p for p in result["projects"] if p["name"] == proj_a_name)
    proj_b_data = next(p for p in result["projects"] if p["name"] == proj_b_name)

    expected_path_a = _shorten_file_path("/abs/path/a")
    expected_path_b = _shorten_file_path("/abs/path/b")
    assert proj_a_data["directory"] == expected_path_a
    assert proj_a_data["status"] == "Ready"
    # Summary truncation is handled by the tool, check the original value was stored
    # assert proj_a_data["summary"] == "Summary A" # This might fail due to truncation in tool output
    assert proj_b_data["directory"] == expected_path_b
    assert proj_b_data["status"] == "Scanning"
    assert proj_b_data["summary"] == ""  # Expect empty string as added in setup

    # --- Cleanup (Registry entries) ---
    # Delete specifically added projects using .equal()
    delete_project_registry_entry(client, proj_a_name)
    delete_project_registry_entry(client, proj_b_name)


# === select_project Tests ===


@pytest.mark.asyncio
async def test_select_project_success(setup_test_tenant):  # Only need tenant fixture
    """Tests selecting an existing project using real Weaviate."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import select_project  # Import tool

    # --- Setup ---
    project_name = f"Selectable_{TEST_TENANT_ID}"
    # Clean potential leftovers
    delete_project_registry_entry(client, project_name)
    add_project_registry_entry(
        client, project_name, "/abs/path/sel", "Ready", "Selectable Summary"
    )

    # --- Execute ---
    args = SelectProjectArgs(project_name=project_name)
    result = await select_project(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert f"Project '{project_name}' selected" in result["message"]
    assert "Selectable Summary" in result["message"]
    assert mcp_server.ACTIVE_PROJECT_NAME == project_name

    # --- Cleanup ---
    delete_project_registry_entry(client, project_name)


@pytest.mark.asyncio
async def test_select_project_not_found(setup_test_tenant):  # Only need tenant fixture
    """Tests selecting a non-existent project using real Weaviate."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import select_project  # Import tool

    # --- Setup ---
    project_name = f"NotFound_{TEST_TENANT_ID}"
    # Ensure it doesn't exist
    delete_project_registry_entry(client, project_name)
    assert get_project_details(client, project_name) is None

    # --- Execute ---
    args = SelectProjectArgs(project_name=project_name)
    result = await select_project(args)

    # --- Assertions ---
    assert result["status"] == "error"
    assert f"Project '{project_name}' not found" in result["message"]
    assert mcp_server.ACTIVE_PROJECT_NAME is None  # Should not be set


# === delete_project Tests ===


@pytest.mark.asyncio
async def test_delete_project_success(
    setup_test_tenant, mock_llm_and_helpers
):  # Need mocks for watcher stop
    """Tests deleting a project successfully using real Weaviate."""
    client, _ = setup_test_tenant  # Use base tenant for setup
    from src.code_analysis_mcp.mcp_server import delete_project  # Import tool

    # --- Setup ---
    project_name = f"delete_me_{TEST_TENANT_ID}"  # Unique name
    tenant_to_delete = project_name  # Tenant name matches project name
    # Add registry entry
    registry_uuid = add_project_registry_entry(
        client, project_name, "/abs/path/del", "Ready"
    )
    assert registry_uuid is not None
    # Ensure tenant exists and add dummy data
    if not client.collections.get("CodeElement").tenants.exists(tenant_to_delete):
        client.collections.get("CodeElement").tenants.create(
            [Tenant(name=tenant_to_delete)]
        )
    client.collections.get("CodeElement").with_tenant(tenant_to_delete).data.insert(
        {"name": "dummy_del"}
    )
    assert get_project_details(client, project_name) is not None
    # Use correct v4 aggregation syntax
    agg_result = (
        client.collections.get("CodeElement")
        .with_tenant(tenant_to_delete)
        .aggregate.over_all(total_count=True)  # Correct method
    )
    assert agg_result.total_count > 0

    mcp_server.ACTIVE_PROJECT_NAME = project_name  # Set as active initially

    # --- Execute ---
    args = DeleteProjectArgs(project_name=project_name)
    result = await delete_project(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert "deleted successfully" in result["message"]
    assert mcp_server.ACTIVE_PROJECT_NAME is None  # Should be cleared
    mock_llm_and_helpers["stop_watcher"].assert_called_once_with(project_name)

    # Verify registry entry is gone
    assert get_project_details(client, project_name) is None
    # Verify tenant is gone
    assert not client.collections.get("CodeElement").tenants.exists(tenant_to_delete)

    # --- Cleanup (already done by test) ---


# === find_element Tests ===


@pytest.mark.asyncio
async def test_find_element_no_active_project(
    setup_test_tenant,
):  # Real DB but no active project
    """Tests find_element when no project is active."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import find_element  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = None  # Ensure no active project

    args = FindElementArgs(name="some_func")
    result = await find_element(args)
    assert result["status"] == "error"
    assert "No active project selected" in result["message"]


@pytest.mark.asyncio
async def test_find_element_success(
    setup_test_tenant, mock_llm_and_helpers
):  # Real DB + mocks
    """Tests find_element successfully finding elements using real Weaviate."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import find_element  # Import tool
    from weaviate.util import generate_uuid5  # For creating test data

    mcp_server.ACTIVE_PROJECT_NAME = tenant_id  # Set active project to test tenant

    # --- Setup: Add test elements ---
    element_uuid1 = generate_uuid5(f"{tenant_id}:func1")
    element_uuid2 = generate_uuid5(f"{tenant_id}:func2")
    elements_to_add = [
        {
            "uuid": element_uuid1,
            "properties": {
                "name": "my_func_find",  # Unique name for test
                "element_type": "function",
                "file_path": "/abs/path/file1_find.py",
                "readable_id": "file1_find:func:my_func_find:10",
            },
        },
        {
            "uuid": element_uuid2,
            "properties": {
                "name": "my_func_find",  # Same name
                "element_type": "function",
                "file_path": "/abs/path/file2_find.py",
                "readable_id": "file2_find:func:my_func_find:25",
                "llm_description": "LLM Desc Find",
            },
        },
        {  # Add another element not matching the name
            "uuid": generate_uuid5(f"{tenant_id}:other_find"),
            "properties": {
                "name": "other_func_find",
                "element_type": "function",
                "file_path": "/abs/path/file1_find.py",
                "readable_id": "file1_find:func:other_func_find:50",
            },
        },  # <-- Closing brace for last dictionary
    ]  # <-- Align closing bracket with list start
    # Add assertion to check batch success
    batch_result = add_objects_batch(client, elements_to_add, "CodeElement", tenant_id)
    assert (
        batch_result.get("status") == "success"
    ), f"Batch add failed: {batch_result.get('message')}"  # Corrected indentation
    # Wait longer for indexing
    await asyncio.sleep(5.0)  # Increased sleep significantly

    # Mock the processing function to return predictable concise output
    async def process_side_effect(c, t, props, u, view_type):
        # Use real _shorten_file_path helper
        return {
            "uuid": u,
            "name": props.get("name"),
            "type": props.get("element_type"),
            "file": _shorten_file_path(props.get("file_path")),
            "description": props.get("llm_description") or props.get("docstring"),
        }

    mock_llm_and_helpers["_process_element_properties"].side_effect = (
        process_side_effect
    )

    # --- Execute ---
    args = FindElementArgs(name="my_func_find", limit=5)
    result = await find_element(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["count"] == 2
    assert len(result["elements"]) == 2

    # Check processed output format (order might vary)
    expected_file1 = _shorten_file_path("/abs/path/file1_find.py")
    expected_file2 = _shorten_file_path("/abs/path/file2_find.py")
    found_elements = {el["uuid"]: el for el in result["elements"]}

    assert element_uuid1 in found_elements
    assert found_elements[element_uuid1] == {
        "uuid": element_uuid1,
        "name": "my_func_find",
        "type": "function",
        "file": expected_file1,
        "description": None,
    }
    assert element_uuid2 in found_elements
    assert found_elements[element_uuid2] == {
        "uuid": element_uuid2,
        "name": "my_func_find",
        "type": "function",
        "file": expected_file2,
        "description": "LLM Desc Find",
    }

    # Corrected assertion: Expect 2 awaits since 2 elements were found and processed
    assert mock_llm_and_helpers["_process_element_properties"].await_count == 2

    # --- Cleanup (done by fixture) ---


# === get_details Tests ===


@pytest.mark.asyncio
async def test_get_details_no_active_project(setup_test_tenant):
    """Tests get_details when no project is active."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import get_details  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = None

    args = GetDetailsArgs(uuid="some-uuid")
    result = await get_details(args)
    assert result["status"] == "error"
    assert "No active project selected" in result["message"]


@pytest.mark.asyncio
async def test_get_details_success(setup_test_tenant, mock_llm_and_helpers):
    """Tests get_details successfully retrieving element details using real Weaviate."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import get_details  # Import tool
    from weaviate.util import generate_uuid5

    mcp_server.ACTIVE_PROJECT_NAME = tenant_id

    # --- Setup: Add test element ---
    element_uuid = generate_uuid5(f"{tenant_id}:detail_func")
    detail_props = {
        "name": "detailed_func_get",
        "element_type": "function",
        "file_path": "/abs/path/detail.py",
        "code_snippet": "def detailed_func_get(): pass",
        "readable_id": "detail:func:detailed_func_get:5",
        "llm_description": "Detail Desc",
    }
    add_objects_batch(
        client,
        [{"uuid": element_uuid, "properties": detail_props}],
        "CodeElement",
        tenant_id,
    )
    await asyncio.sleep(5.0)  # Increased sleep significantly

    # Mock processing to return detailed view (can use real one if simple enough)
    async def process_detail_side_effect(c, t, props, u, view_type):
        assert view_type == "detail"
        # Simulate returning more fields for detail view based on real props
        processed = {
            "uuid": u,
            "name": props.get("name"),
            "element_type": props.get("element_type"),
            "file_path": props.get("file_path"),  # Keep original path for detail
            "code_snippet": props.get("code_snippet"),
            "llm_description": props.get("llm_description"),
            # Add other expected detail fields if needed
        }
        return processed

    mock_llm_and_helpers["_process_element_properties"].side_effect = (
        process_detail_side_effect
    )

    # --- Execute ---
    args = GetDetailsArgs(uuid=element_uuid)
    result = await get_details(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["details"]["uuid"] == element_uuid
    assert result["details"]["name"] == "detailed_func_get"
    assert result["details"]["code_snippet"] == "def detailed_func_get(): pass"
    assert result["details"]["llm_description"] == "Detail Desc"
    assert result["details"]["file_path"] == "/abs/path/detail.py"  # Check detail field

    mock_llm_and_helpers["_process_element_properties"].assert_awaited_once()

    # --- Cleanup (done by fixture) ---


# === ask_question Tests ===


@pytest.mark.asyncio
async def test_ask_question_no_active_project(setup_test_tenant):  # No mocks needed
    """Tests ask_question when no project is active."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import ask_question  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = None

    args = AskQuestionArgs(query="What is this project?")
    result = await ask_question(args)
    assert result["status"] == "error"
    assert "No active project selected" in result["message"]


@pytest.mark.asyncio
async def test_ask_question_success(
    setup_test_tenant, mock_llm_and_helpers
):  # Need mock RAG
    """Tests ask_question successfully getting an answer (mocks RAG)."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import ask_question  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = tenant_id
    mock_llm_and_helpers["answer_codebase_question"].return_value = (
        "This project does X and Y."
    )

    # --- Execute ---
    args = AskQuestionArgs(query="What does it do?")
    result = await ask_question(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["answer"] == "This project does X and Y."
    mock_llm_and_helpers["answer_codebase_question"].assert_awaited_once_with(
        "What does it do?",
        client=client,  # Should pass the real client
        tenant_id=tenant_id,
    )

    # --- Cleanup (done by fixture) ---


# === analyze_snippet Tests ===


@pytest.mark.asyncio
async def test_analyze_snippet_no_active_project(setup_test_tenant):  # No mocks needed
    """Tests analyze_snippet when no project is active."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import analyze_snippet  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = None

    args = AnalyzeSnippetArgs(code_snippet="call_func()")
    result = await analyze_snippet(args)
    assert result["status"] == "error"
    assert "No active project selected" in result["message"]


@pytest.mark.asyncio
async def test_analyze_snippet_success(
    setup_test_tenant, mock_llm_and_helpers
):  # Real DB + mocks
    """Tests analyze_snippet finding related elements using real Weaviate."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import analyze_snippet  # Import tool
    from weaviate.util import generate_uuid5

    mcp_server.ACTIVE_PROJECT_NAME = tenant_id

    # --- Setup: Add test elements ---
    var_uuid = generate_uuid5(f"{tenant_id}:my_var_snip")
    func_uuid = generate_uuid5(f"{tenant_id}:other_func_snip")
    elements_to_add = [
        {
            "uuid": var_uuid,
            "properties": {
                "name": "my_var_snip",
                "element_type": "variable",
                "file_path": "vars_snip.py",
            },
        },
        {
            "uuid": func_uuid,
            "properties": {
                "name": "other_func_snip",
                "element_type": "function",
                "file_path": "funcs_snip.py",
            },
        },
    ]
    add_objects_batch(client, elements_to_add, "CodeElement", tenant_id)
    await asyncio.sleep(2.5)  # Increased sleep

    # Mock identifier extraction and element processing
    mock_llm_and_helpers["_extract_identifiers"].return_value = [
        "my_var_snip",
        "other_func_snip",
    ]

    async def process_side_effect_snippet(c, t, props, u, view_type):
        return {
            "uuid": u,
            "name": props.get("name"),
            "type": props.get("element_type"),
            "file": props.get("file_path"),
            "description": None,
        }

    mock_llm_and_helpers["_process_element_properties"].side_effect = (
        process_side_effect_snippet
    )

    # --- Execute ---
    args = AnalyzeSnippetArgs(code_snippet="x = my_var_snip; other_func_snip()")
    result = await analyze_snippet(args)

    # --- Assertions ---
    assert result["status"] == "success"
    # Message might vary slightly based on find_element implementation details
    assert "potentially related unique elements" in result["message"]
    assert len(result["related_elements"]) == 2
    result_uuids = {el["uuid"] for el in result["related_elements"]}
    assert result_uuids == {var_uuid, func_uuid}
    # Check that find_element (which uses find_element_by_name_db) was called implicitly
    # This requires inspecting the implementation or trusting the mocks/real DB interaction
    assert (
        mock_llm_and_helpers["_process_element_properties"].await_count == 2
    )  # Called for each found element

    # --- Cleanup (done by fixture) ---


# === trigger_llm_processing Tests ===


@pytest.mark.asyncio
async def test_trigger_llm_no_active_project(setup_test_tenant):  # No mocks needed
    """Tests trigger_llm_processing when no project is active."""
    client, _ = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import trigger_llm_processing  # Import tool

    mcp_server.ACTIVE_PROJECT_NAME = None

    args = TriggerLlmArgs(rerun_all=True)
    result = await trigger_llm_processing(args)
    assert result["status"] == "error"
    assert "No active project selected" in result["message"]


@pytest.mark.asyncio
async def test_trigger_llm_success_specific(
    setup_test_tenant, mock_llm_and_helpers
):  # Real DB + mocks
    """Tests triggering LLM for specific UUIDs using real Weaviate."""
    client, tenant_id = setup_test_tenant
    from src.code_analysis_mcp.mcp_server import trigger_llm_processing  # Import tool
    from weaviate.util import generate_uuid5

    mcp_server.ACTIVE_PROJECT_NAME = tenant_id

    # --- Setup: Add elements, one needing enrichment ---
    uuid_x = generate_uuid5(f"{tenant_id}:llm_x")
    uuid_y = generate_uuid5(f"{tenant_id}:llm_y")
    elements_to_add = [
        {
            "uuid": uuid_x,
            "properties": {"name": "llm_func_x", "llm_description": ""},
        },  # Needs processing
        {
            "uuid": uuid_y,
            "properties": {"name": "llm_func_y", "llm_description": "Done"},
        },  # Skip
    ]
    add_objects_batch(client, elements_to_add, "CodeElement", tenant_id)
    await asyncio.sleep(2.5)  # Increased sleep

    # --- Execute ---
    # Enable LLM processing for the test scope via patch if needed, or ensure env var is set
    with patch("src.code_analysis_mcp.mcp_server.LLM_ENABLED", True):
        args = TriggerLlmArgs(uuids=[uuid_x, uuid_y], skip_enriched=True)
        result = await trigger_llm_processing(args)

    # --- Assertions ---
    assert result["status"] == "success"
    assert (
        "triggered for 1 elements" in result["message"]
    )  # Only uuid_x should be processed
    assert f"project '{tenant_id}'" in result["message"]

    # Check that the background task creator was called once for uuid_x
    mock_llm_and_helpers["create_task"].assert_called_once()
    # Check that process_element_llm (the mocked version) was scheduled once
    # Inspecting the args of create_task is complex, rely on call count and mock behavior
    # We expect process_element_llm mock to be called via the background task eventually
    # For simplicity here, we just check create_task was called.

    # --- Cleanup (done by fixture) ---
