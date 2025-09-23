import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock

import src.code_analysis_mcp.mcp_server as mcp_server
from src.code_analysis_mcp.codebase_manager import CodebaseManager
from src.code_analysis_mcp.weaviate_client import create_schema, delete_tenant, get_codebase_details
import weaviate

# --- Constants ---
TEST_TENANT_ID = "_pytest_integration_tenant_"

# --- Fixtures ---

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def real_weaviate_client():
    """Creates a real Weaviate client connection for the module."""
    client = weaviate.connect_to_local()
    client.connect()
    create_schema(client)
    yield client
    client.close()

@pytest.fixture(scope="function")
async def mcp_server_setup(real_weaviate_client):
    """
    Simulates the lifespan of the MCP server for integration tests.
    Provides a ready-to-use CodebaseManager instance.
    """
    # Set the real client on the mcp_server module for the tools to use
    manager = CodebaseManager()
    manager.client = real_weaviate_client
    mcp_server.codebase_manager = manager

    yield manager

    # Teardown logic if any
    mcp_server.codebase_manager = None

# --- Integration Tests ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_scan_select_find_integration(mcp_server_setup):
    """
    An end-to-end integration test for scanning, selecting, and finding an element.
    """
    manager = mcp_server_setup
    codebase_name = f"{TEST_TENANT_ID}_{os.urandom(4).hex()}"

    # Create a dummy directory and file to scan
    dummy_dir = f"/tmp/{codebase_name}"
    os.makedirs(dummy_dir, exist_ok=True)
    with open(f"{dummy_dir}/app.py", "w") as f:
        f.write("def my_awesome_function():\n    print('hello world')\n")

    try:
        # 1. Scan the codebase
        scan_result = await mcp_server.scan_codebase(codebase_name=codebase_name, directory=dummy_dir)
        assert scan_result["status"] == "success"
        assert manager.active_codebase_name == codebase_name

        # Allow time for indexing
        await asyncio.sleep(2)

        # Verify that the codebase is in the registry
        details = get_codebase_details(manager.client, codebase_name)
        assert details is not None
        assert details["status"] in ["Summarizing", "Ready"] # Can be either depending on timing

        # 2. Reset and Select the codebase
        manager.active_codebase_name = None
        select_result = await mcp_server.select_codebase(codebase_name=codebase_name)
        assert select_result["status"] == "success"
        assert manager.active_codebase_name == codebase_name

        # 3. Find the function
        find_result = await mcp_server.find_element(name="my_awesome_function")
        assert find_result["status"] == "success"
        assert find_result["count"] == 1
        assert find_result["elements"][0]["name"] == "my_awesome_function"

    finally:
        # Cleanup
        await mcp_server.delete_codebase(codebase_name=codebase_name)
        os.remove(f"{dummy_dir}/app.py")
        os.rmdir(dummy_dir)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_codebase_integration(mcp_server_setup):
    """
    Tests that deleting a codebase correctly removes it from the registry and cleans up the tenant.
    """
    manager = mcp_server_setup
    codebase_name = f"{TEST_TENANT_ID}_{os.urandom(4).hex()}"

    # Setup: Scan a codebase to create it
    dummy_dir = f"/tmp/{codebase_name}"
    os.makedirs(dummy_dir, exist_ok=True)
    with open(f"{dummy_dir}/temp.py", "w") as f:
        f.write("pass")

    await mcp_server.scan_codebase(codebase_name=codebase_name, directory=dummy_dir)
    assert get_codebase_details(manager.client, codebase_name) is not None
    assert manager.client.collections.get("CodeElement").tenants.exists(codebase_name)

    # Execute delete
    delete_result = await mcp_server.delete_codebase(codebase_name=codebase_name)
    assert delete_result["status"] == "success"

    # Verify cleanup
    assert get_codebase_details(manager.client, codebase_name) is None
    assert not manager.client.collections.get("CodeElement").tenants.exists(codebase_name)

    # Cleanup dummy dir
    os.remove(f"{dummy_dir}/temp.py")
    os.rmdir(dummy_dir)
