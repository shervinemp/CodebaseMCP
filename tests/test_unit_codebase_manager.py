import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.code_analysis_mcp.codebase_manager import CodebaseManager
from src.code_analysis_mcp.weaviate_client import create_schema, delete_tenant
import weaviate
import os

# --- Constants ---
TEST_TENANT_ID = "_pytest_tenant_"

# --- Fixtures ---

@pytest.fixture
def mock_weaviate_client():
    """Mocks the Weaviate client for unit tests."""
    mock_client = MagicMock(spec=weaviate.Client)
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_client.collections = mock_collections

    # Mock tenant existence
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True

    # Mock collection gets
    mock_code_element_collection = MagicMock()
    mock_code_element_collection.tenants = mock_tenants
    mock_code_file_collection = MagicMock()
    mock_code_file_collection.tenants = mock_tenants
    mock_registry_collection = MagicMock()

    mock_collections.get.side_effect = lambda name: {
        "CodeElement": mock_code_element_collection,
        "CodeFile": mock_code_file_collection,
        "CodebaseRegistry": mock_registry_collection
    }[name]

    return mock_client

@pytest.fixture
async def initialized_manager_unit():
    """Provides an initialized CodebaseManager with a mocked client for unit tests."""
    manager = CodebaseManager()
    with patch('src.code_analysis_mcp.codebase_manager.create_weaviate_client', new_callable=MagicMock) as mock_create_client:
        mock_client = mock_weaviate_client()
        mock_create_client.return_value = mock_client

        # Mock the async to_thread call for schema creation
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            await manager.initialize()
            mock_to_thread.assert_awaited_with(create_schema, mock_client)
            yield manager

# --- Unit Tests ---

@pytest.mark.unit
@pytest.mark.asyncio
async def test_select_codebase_unit(initialized_manager_unit: CodebaseManager):
    """Unit test for selecting a codebase."""
    manager = initialized_manager_unit
    codebase_name = "test_codebase"

    # Mock the get_codebase_details function
    with patch('src.code_analysis_mcp.codebase_manager.get_codebase_details') as mock_get_details:
        mock_get_details.return_value = {"summary": "A test summary."}

        result = manager.select_codebase(codebase_name)

        assert result["status"] == "success"
        assert manager.active_codebase_name == codebase_name
        assert "A test summary." in result["message"]
        mock_get_details.assert_called_once_with(manager.client, codebase_name)

@pytest.mark.unit
@pytest.mark.asyncio
@patch('os.path.isdir', return_value=True)
async def test_scan_codebase_unit(mock_isdir, initialized_manager_unit: CodebaseManager):
    """Unit test for scanning a new codebase."""
    manager = initialized_manager_unit
    codebase_name = "new_scanned_codebase"
    directory = "/fake/dir"

    with patch('src.code_analysis_mcp.codebase_manager.get_codebase_details', return_value=None) as mock_get_details, \
         patch('src.code_analysis_mcp.codebase_manager.add_codebase_registry_entry', return_value="uuid-123") as mock_add_entry, \
         patch('src.code_analysis_mcp.codebase_manager._scan_cleanup_and_upload', new_callable=AsyncMock, return_value=("OK", ["uuid-abc"])) as mock_scan, \
         patch('src.code_analysis_mcp.codebase_manager.CodebaseManager.background_generate_summary', new_callable=AsyncMock) as mock_summary, \
         patch('src.code_analysis_mcp.codebase_manager.start_watcher', return_value=(True, "Watcher started.")) as mock_start_watcher:

        result = await manager.scan_codebase(codebase_name, directory)

        assert result["status"] == "success"
        assert codebase_name in result["message"]
        mock_get_details.assert_called_once()
        mock_add_entry.assert_called_once()
        mock_scan.assert_awaited_once()
        mock_summary.assert_awaited_once()
        mock_start_watcher.assert_called_once()

# --- Integration Tests ---

# Fixture for a real weaviate client, module-scoped
@pytest.fixture(scope="module")
def real_weaviate_client():
    client = weaviate.connect_to_local()
    client.connect()
    create_schema(client)
    yield client
    client.close()

@pytest.fixture(scope="function")
def integration_setup(real_weaviate_client):
    """Sets up and tears down a tenant for integration tests."""
    client = real_weaviate_client
    tenant_id = f"{TEST_TENANT_ID}_{os.urandom(4).hex()}" # Unique tenant for each test

    for collection_name in ["CodeElement", "CodeFile"]:
        collection = client.collections.get(collection_name)
        if not collection.tenants.exists(tenant_id):
            collection.tenants.create([weaviate.classes.tenants.Tenant(name=tenant_id)])

    yield client, tenant_id

    # Teardown
    delete_tenant(client, tenant_id)


@pytest.fixture
async def initialized_manager_integration(integration_setup):
    """Provides an initialized CodebaseManager with a real client for integration tests."""
    client, tenant_id = integration_setup
    manager = CodebaseManager()

    # We can't easily patch create_weaviate_client here, so we set the client directly
    # This is a small compromise for testing.
    manager.client = client

    # Since client is already connected and schema exists, initialize is simpler
    # We can assume it works based on the fixture setup.
    # await manager.initialize() # This would try to recreate client/schema

    yield manager, client, tenant_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scan_and_select_codebase_integration(initialized_manager_integration):
    """Full integration test for scanning and then selecting a codebase."""
    manager, client, tenant_id = initialized_manager_integration

    # Use the unique tenant_id as the codebase_name
    codebase_name = tenant_id

    # Create a dummy directory and file to scan
    dummy_dir = f"/tmp/{codebase_name}"
    os.makedirs(dummy_dir, exist_ok=True)
    with open(f"{dummy_dir}/test_module.py", "w") as f:
        f.write("def hello():\n    print('world')\n")

    # 1. Scan the codebase
    scan_result = await manager.scan_codebase(codebase_name, dummy_dir)

    assert scan_result["status"] == "success", f"Scan failed: {scan_result.get('message')}"
    assert manager.active_codebase_name == codebase_name

    # Give weaviate a moment to index
    await asyncio.sleep(2)

    # 2. Select the codebase
    manager.active_codebase_name = None # Reset active codebase
    select_result = manager.select_codebase(codebase_name)

    assert select_result["status"] == "success"
    assert manager.active_codebase_name == codebase_name

    # 3. Find an element from the scan
    find_result = await manager.find_element(name="hello")
    assert find_result["status"] == "success"
    assert find_result["count"] == 1
    assert find_result["elements"][0]["name"] == "hello"

    # Cleanup the dummy directory
    os.remove(f"{dummy_dir}/test_module.py")
    os.rmdir(dummy_dir)
