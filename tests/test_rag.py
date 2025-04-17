import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
import google.api_core.exceptions
import sys
import os

# Update imports to reflect new structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.code_analysis_mcp.mcp_server as mcp_server  # Keep import for clearing background tasks

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


try:
    # Import new summary function as well
    from src.code_analysis_mcp.rag import (
        answer_codebase_question,
        refine_element_description,
        generate_codebase_summary,
    )
    from src.code_analysis_mcp.code_scanner import (
        enrich_element,
    )  # Keep for now, though tests are removed
except ImportError:
    pytest.skip(
        "Could not import RAG/Scanner module or function", allow_module_level=True
    )

    # Add dummy for new function
    async def generate_codebase_summary(*args, **kwargs):
        return "IMPORT ERROR"

    async def answer_codebase_question(*args, **kwargs):
        return "IMPORT ERROR"

    async def refine_element_description(*args, **kwargs):
        return False

    async def enrich_element(*args, **kwargs):
        return False


@pytest.fixture(autouse=True)
def clear_background_tasks():
    """Ensures the background task set is clear before/after rag tests."""
    # Use the imported mcp_server module
    original_tasks = set(mcp_server.background_llm_tasks)
    mcp_server.background_llm_tasks.clear()
    yield
    tasks_to_cancel = list(mcp_server.background_llm_tasks)
    if tasks_to_cancel:
        for task in tasks_to_cancel:
            if not task.done():  # Check if task is done before cancelling
                task.cancel()
    mcp_server.background_llm_tasks.clear()


# Helper to create mock Weaviate Object
def create_mock_weaviate_object(uuid, properties, vector=None, references=None):
    mock_obj = MagicMock()
    mock_obj.uuid = uuid
    mock_obj.properties = properties
    mock_obj.vector = vector
    mock_obj.references = references or {}
    return mock_obj


# --- Tests for answer_codebase_question ---


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.create_weaviate_client")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.find_relevant_elements", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.embedding_model_name", "mock-embedding-model"
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_rag_answer_generation_success(
    mock_generate_content, mock_find_elements, mock_create_client
):
    """Tests successful answer generation when context is found."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    # Mock tenant existence check
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client

    mock_find_elements.return_value = [
        {
            "uuid": "uuid-h1",  # Need UUID for processing
            "properties": {
                "file_path": "test.py",
                "element_type": "function",
                "name": "hello",
                "code_snippet": "def hello():\n  print('world')",
                "llm_description": "Prints world",
            },
        }
    ]
    mock_generate_response = MagicMock()
    mock_generate_response.text = "The function hello prints world."
    mock_generate_content.return_value = mock_generate_response

    # Provide tenant_id
    answer = await answer_codebase_question(
        "What does hello function do?", client=mock_client, tenant_id="TestTenantRAG"
    )

    assert "The function hello prints world." in answer
    mock_find_elements.assert_awaited_once_with(
        mock_client,
        "TestTenantRAG",
        "What does hello function do?",
        limit=3,  # Check tenant_id passed
    )
    mock_generate_content.assert_awaited_once()
    # Check prompt includes tenant_id/project name
    prompt_arg = mock_generate_content.call_args.args[0]
    assert "project named 'TestTenantRAG'" in prompt_arg


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.create_weaviate_client")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.find_relevant_elements", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.embedding_model_name", "mock-embedding-model"
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_rag_no_context_found(
    mock_generate_content, mock_find_elements, mock_create_client
):
    """Tests the response when no relevant context is found."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client
    mock_find_elements.return_value = []

    # Provide tenant_id
    answer = await answer_codebase_question(
        "Tell me about elephants?", client=mock_client, tenant_id="TestTenantNoCtx"
    )

    assert "Could not find relevant context" in answer
    mock_find_elements.assert_awaited_once_with(
        mock_client,
        "TestTenantNoCtx",
        "Tell me about elephants?",
        limit=3,  # Check tenant_id
    )
    mock_generate_content.assert_not_awaited()


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.create_weaviate_client")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.find_relevant_elements", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.embedding_model_name", "mock-embedding-model"
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_rag_llm_generation_error(
    mock_generate_content, mock_find_elements, mock_create_client
):
    """Tests error handling during LLM generation."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client
    mock_find_elements.return_value = [
        {"uuid": "uuid-e1", "properties": {"code_snippet": "context"}}
    ]
    mock_generate_content.side_effect = Exception("LLM API Failed")

    # Provide tenant_id
    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id="TestTenantLLMErr"
    )

    assert "ERROR: Failed to generate answer using LLM" in answer
    assert "LLM API Failed" in answer
    mock_find_elements.assert_awaited_once()
    mock_generate_content.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.create_weaviate_client")  # Update patch target
async def test_rag_tenant_not_provided(mock_create_client):
    """Tests error when tenant_id is not provided."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_create_client.return_value = mock_client

    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id=None
    )  # No tenant_id
    assert "ERROR: tenant_id must be provided" in answer


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.create_weaviate_client")  # Update patch target
async def test_rag_tenant_does_not_exist(mock_create_client):
    """Tests error when the specified tenant does not exist."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = False  # Tenant does NOT exist
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client

    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id="MissingTenant"
    )
    assert "ERROR: Project (tenant) 'MissingTenant' does not exist" in answer
    mock_tenants.exists.assert_called_once_with("MissingTenant")


# --- Tests for refine_element_description ---

# Note: enrich_element tests removed as the function is in code_scanner now,
# and these tests focus on the RAG module's functions.


@pytest.mark.asyncio
@patch("src.code_analysis_mcp.rag.get_element_details")  # Update patch target
@patch("src.code_analysis_mcp.rag.update_element_properties")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.find_element_by_name")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.find_relevant_elements", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.wvc_query")  # Update patch target
@patch(
    "src.code_analysis_mcp.rag.asyncio.sleep", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_refine_success(
    mock_sleep,
    mock_wvc_query,
    mock_find_relevant_elements,
    mock_find_element_by_name,
    mock_generate_content,
    mock_update_element_properties,
    mock_get_element_details,
):
    """Tests successful refinement of a function description."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()  # Mock the object returned by with_tenant
    mock_collections.get.return_value = mock_collections
    mock_collections.with_tenant.return_value = (
        mock_tenant_collection  # Mock with_tenant
    )
    mock_collections.tenants.exists.return_value = True  # Assume tenant exists

    element_uuid = "func-uuid-1"
    tenant_id = "TestTenantRefine"
    initial_props = {
        "name": "test_func",
        "element_type": "function",
        "code_snippet": "def test_func(): pass",
        "llm_description": "Initial desc",
        "readable_id": "test.py:function:test_func:1",
        "signature": "test_func()",
        "file_path": "test.py",  # Needed for sibling search
    }
    mock_element = create_mock_weaviate_object(element_uuid, initial_props)
    mock_get_element_details.return_value = mock_element

    # Mock the chained calls for fetching references and objects
    mock_tenant_collection.query.fetch_object_by_id.return_value = (
        create_mock_weaviate_object(element_uuid, initial_props, references={})
    )
    mock_tenant_collection.query.fetch_objects.return_value = MagicMock(objects=[])
    mock_find_element_by_name.return_value = []  # No siblings found
    mock_find_relevant_elements.return_value = []  # No related vars found

    mock_generate_response = MagicMock()
    mock_generate_response.text = "Refined description."
    mock_generate_content.return_value = mock_generate_response

    mock_update_element_properties.return_value = True

    # Provide tenant_id
    success = await refine_element_description(mock_client, tenant_id, element_uuid)

    assert success is True
    mock_get_element_details.assert_called_once_with(
        mock_client, tenant_id, element_uuid
    )  # Check tenant_id
    mock_generate_content.assert_awaited_once()
    # Check prompt includes tenant_id/project name
    prompt_arg = mock_generate_content.call_args.args[0]
    assert f"project '{tenant_id}'" in prompt_arg

    expected_updated_props = initial_props.copy()
    expected_updated_props["llm_description"] = "Refined description."
    mock_update_element_properties.assert_called_once()
    args, kwargs = mock_update_element_properties.call_args
    assert args[0] == mock_client
    assert args[1] == tenant_id  # Check tenant_id
    assert args[2] == element_uuid
    assert args[3] == expected_updated_props  # Properties dict is 4th arg
    mock_sleep.assert_not_called()


# Add more refine tests, ensuring tenant_id is passed and checked where appropriate...
# e.g., test_refine_skip_non_function, test_refine_rate_limit_retry_success, etc.
# Make sure mocks like get_element_details, find_element_by_name, update_element_properties
# are called with the correct tenant_id.

# --- Tests for generate_codebase_summary ---


@pytest.mark.asyncio
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_generate_summary_success(mock_generate_content):
    """Tests successful project summary generation."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    # Correctly mock the chain: client -> collections -> get -> tenants -> exists
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()  # Mock for the 'CodeElement' collection
    mock_element_collection_mock.tenants = mock_tenants  # Assign tenants mock here
    mock_element_collection_mock.with_tenant.return_value = (
        mock_tenant_collection  # Make with_tenant return the correct mock
    )
    mock_collections.get.return_value = mock_element_collection_mock  # Make collections.get return the specific collection mock
    mock_client.collections = mock_collections  # Assign collections mock to client
    mock_tenant_collection.query = (
        mock_query  # Assign query mock to the tenant collection mock
    )

    # Mock elements found in the tenant
    mock_func = create_mock_weaviate_object(
        "uuid-fn",
        {
            "name": "main_func",
            "element_type": "function",
            "file_path": "main.py",
            "llm_description": "Does main things.",
        },
    )
    mock_class = create_mock_weaviate_object(
        "uuid-cl",
        {
            "name": "HelperClass",
            "element_type": "class",
            "file_path": "helpers.py",
            "docstring": "A helper.",
        },
    )
    mock_query.fetch_objects.return_value = MagicMock(objects=[mock_func, mock_class])

    mock_generate_response = MagicMock()
    mock_generate_response.text = (
        "This project contains a main function and a helper class."
    )
    mock_generate_content.return_value = mock_generate_response

    project_name = "SummaryProject"
    summary = await generate_codebase_summary(mock_client, project_name)

    assert summary == "This project contains a main function and a helper class."
    # Assert against the specific mock object
    mock_tenants.exists.assert_called_once_with(project_name)
    mock_query.fetch_objects.assert_called_once()
    mock_generate_content.assert_awaited_once()
    # Check prompt includes project name and context
    prompt_arg = mock_generate_content.call_args.args[0]
    assert f"project named '{project_name}'" in prompt_arg
    assert "main_func" in prompt_arg
    assert "HelperClass" in prompt_arg


@pytest.mark.asyncio
@patch(
    "src.code_analysis_mcp.rag.model.generate_content",
    new_callable=AsyncMock,  # Update patch target
)  # Still need to mock generate_content even if not called
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_generate_summary_tenant_not_found(
    mock_generate_content,
):  # Add mock param
    """Tests summary generation when the tenant doesn't exist."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()  # Create tenants mock
    mock_tenants.exists.return_value = False  # Tenant does not exist
    mock_collections.get.return_value.tenants = mock_tenants  # Assign tenants mock
    mock_client.collections = mock_collections

    project_name = "MissingSummaryProject"
    summary = await generate_codebase_summary(
        mock_client, project_name
    )  # Await the coroutine

    assert "Error: Project 'MissingSummaryProject' not found" in summary
    mock_tenants.exists.assert_called_once_with(
        project_name
    )  # Assert on the specific mock
    mock_generate_content.assert_not_awaited()  # Ensure LLM not called


@pytest.mark.asyncio
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_generate_summary_no_elements(mock_generate_content):
    """Tests summary generation when no key elements are found."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    # Correctly mock the chain: client -> collections -> get -> tenants -> exists
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()  # Mock for the 'CodeElement' collection
    mock_element_collection_mock.tenants = mock_tenants  # Assign tenants mock here
    mock_element_collection_mock.with_tenant.return_value = (
        mock_tenant_collection  # Make with_tenant return the correct mock
    )
    mock_collections.get.return_value = mock_element_collection_mock  # Make collections.get return the specific collection mock
    mock_client.collections = mock_collections  # Assign collections mock to client
    mock_tenant_collection.query = mock_query
    mock_query.fetch_objects.return_value = MagicMock(objects=[])  # No objects found

    project_name = "EmptySummaryProject"
    summary = await generate_codebase_summary(
        mock_client, project_name
    )  # Await the coroutine

    assert "no key functions or classes found" in summary
    mock_query.fetch_objects.assert_called_once()
    mock_generate_content.assert_not_awaited()


@pytest.mark.asyncio
@patch(
    "src.code_analysis_mcp.rag.model.generate_content", new_callable=AsyncMock
)  # Update patch target
@patch("src.code_analysis_mcp.rag.model", MagicMock())  # Update patch target
async def test_generate_summary_llm_error(mock_generate_content):
    """Tests summary generation when the LLM call fails."""
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    # Correctly mock the chain: client -> collections -> get -> tenants -> exists
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()  # Mock for the 'CodeElement' collection
    mock_element_collection_mock.tenants = mock_tenants  # Assign tenants mock here
    mock_element_collection_mock.with_tenant.return_value = (
        mock_tenant_collection  # Make with_tenant return the correct mock
    )
    mock_collections.get.return_value = mock_element_collection_mock  # Make collections.get return the specific collection mock
    mock_client.collections = mock_collections  # Assign collections mock to client
    mock_tenant_collection.query = mock_query
    mock_func = create_mock_weaviate_object(
        "uuid-fn", {"name": "main_func", "element_type": "function"}
    )
    mock_query.fetch_objects.return_value = MagicMock(objects=[mock_func])

    mock_generate_content.side_effect = google.api_core.exceptions.InternalServerError(
        "LLM Down"
    )

    project_name = "LLMSummaryFail"
    summary = await generate_codebase_summary(
        mock_client, project_name
    )  # Await the coroutine

    assert "Error generating summary" in summary
    assert "LLM Down" in summary
    mock_query.fetch_objects.assert_called_once()
    mock_generate_content.assert_awaited_once()
