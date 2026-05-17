import pytest
from unittest.mock import MagicMock, patch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.code_analysis_mcp.rag import (
        answer_codebase_question,
        refine_element_description,
        generate_codebase_summary,
    )
except ImportError as e:
    pytest.skip(f"Could not import RAG/Scanner module: {e}", allow_module_level=True)


# --- Tests for include_dependencies on ask_question ---


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@patch("src.code_analysis_mcp.rag.find_relevant_elements")
@pytest.mark.asyncio
async def test_rag_excludes_dependencies(
    mock_find_elements, mock_create_client, mock_get_provider
):
    """When include_dependencies=False, only the primary tenant is queried."""
    mock_get_provider.return_value = create_mock_provider("answer")
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client
    mock_find_elements.return_value = [
        {"uuid": "x", "properties": {"code_snippet": "def foo(): pass"}}
    ]

    answer = await answer_codebase_question(
        "test?", client=mock_client, tenant_id="Primary", include_dependencies=False
    )

    assert "answer" in answer
    # Should not have fetched deps — get_codebase_details should not be called
    # find_relevant_elements should be called with only [Primary]
    call_args = mock_find_elements.call_args
    assert call_args is not None
    assert call_args[0][1] == ["Primary"]  # tenant_ids


# --- Tests for include_dependencies on ask_question ---


def create_mock_weaviate_object(uuid, properties, vector=None, references=None):
    mock_obj = MagicMock()
    mock_obj.uuid = uuid
    mock_obj.properties = properties
    mock_obj.vector = vector
    mock_obj.references = references or {}
    return mock_obj


def create_mock_provider(return_text="Generated response."):
    mock_provider = MagicMock()
    mock_provider.is_available = True
    mock_provider.name = "test-provider"
    mock_provider.generate.return_value = return_text
    mock_provider.embed.return_value = [0.1, 0.2, 0.3]
    return mock_provider


# --- Tests for answer_codebase_question ---


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@patch("src.code_analysis_mcp.rag.find_relevant_elements")
@pytest.mark.asyncio
async def test_rag_answer_generation_success(
    mock_find_elements, mock_create_client, mock_get_provider
):
    """Tests successful answer generation when context is found."""
    mock_get_provider.return_value = create_mock_provider(
        "The function hello prints world."
    )
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client
    mock_find_elements.return_value = [
        {
            "uuid": "uuid-h1",
            "properties": {
                "file_path": "test.py",
                "element_type": "function",
                "name": "hello",
                "code_snippet": "def hello():\n  print('world')",
                "llm_description": "Prints world",
            },
        }
    ]

    answer = await answer_codebase_question(
        "What does hello function do?", client=mock_client, tenant_id="TestTenantRAG"
    )

    assert "The function hello prints world." in answer
    mock_find_elements.assert_called_once()
    mock_get_provider.assert_called_once()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@patch("src.code_analysis_mcp.rag.find_relevant_elements")
@pytest.mark.asyncio
async def test_rag_no_context_found(
    mock_find_elements, mock_create_client, mock_get_provider
):
    """Tests the response when no relevant context is found."""
    mock_get_provider.return_value = create_mock_provider()
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client
    mock_find_elements.return_value = []

    answer = await answer_codebase_question(
        "Tell me about elephants?", client=mock_client, tenant_id="TestTenantNoCtx"
    )

    assert "Could not find relevant context" in answer
    mock_find_elements.assert_called_once()
    mock_get_provider.assert_called_once()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@patch("src.code_analysis_mcp.rag.find_relevant_elements")
@pytest.mark.asyncio
async def test_rag_llm_generation_error(
    mock_find_elements, mock_create_client, mock_get_provider
):
    """Tests error handling during LLM generation."""
    from src.code_analysis_mcp.llm import LLMProviderError

    mock_provider = create_mock_provider()
    mock_provider.generate.side_effect = LLMProviderError("LLM API Failed")
    mock_get_provider.return_value = mock_provider
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

    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id="TestTenantLLMErr"
    )

    assert "ERROR: LLM provider error during generation" in answer
    assert "LLM API Failed" in answer
    mock_find_elements.assert_called_once()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@pytest.mark.asyncio
async def test_rag_tenant_not_provided(mock_create_client, mock_get_provider):
    """Tests error when tenant_id is not provided."""
    mock_get_provider.return_value = create_mock_provider()
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_create_client.return_value = mock_client

    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id=None
    )
    assert "ERROR: tenant_id must be provided" in answer


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.create_weaviate_client")
@pytest.mark.asyncio
async def test_rag_tenant_does_not_exist(mock_create_client, mock_get_provider):
    """Tests error when the specified tenant does not exist."""
    mock_get_provider.return_value = create_mock_provider()
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = False
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections
    mock_create_client.return_value = mock_client

    answer = await answer_codebase_question(
        "A question?", client=mock_client, tenant_id="MissingTenant"
    )
    assert "ERROR: Codebase (tenant) 'MissingTenant' does not exist" in answer


# --- Tests for refine_element_description ---


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@patch("src.code_analysis_mcp.rag.get_element_details")
@patch("src.code_analysis_mcp.rag.update_element_properties")
@patch("src.code_analysis_mcp.rag.find_element_by_name")
@patch("src.code_analysis_mcp.rag.find_relevant_elements")
@patch("src.code_analysis_mcp.rag.wvc_query")
@patch("src.code_analysis_mcp.rag.asyncio.sleep", new_callable=MagicMock)
@pytest.mark.asyncio
async def test_refine_success(
    mock_sleep,
    mock_wvc_query,
    mock_find_relevant_elements,
    mock_find_element_by_name,
    mock_update_element_properties,
    mock_get_element_details,
    mock_get_provider,
):
    """Tests successful refinement of a function description."""
    mock_get_provider.return_value = create_mock_provider("Refined description.")
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_collections.get.return_value = mock_collections
    mock_collections.with_tenant.return_value = mock_tenant_collection
    mock_collections.tenants.exists.return_value = True

    element_uuid = "func-uuid-1"
    tenant_id = "TestTenantRefine"
    initial_props = {
        "name": "test_func",
        "element_type": "function",
        "code_snippet": "def test_func(): pass",
        "llm_description": "Initial desc",
        "readable_id": "test.py:function:test_func:1",
        "signature": "test_func()",
        "file_path": "test.py",
    }
    mock_element = create_mock_weaviate_object(element_uuid, initial_props)
    mock_get_element_details.return_value = mock_element
    mock_tenant_collection.query.fetch_object_by_id.return_value = (
        create_mock_weaviate_object(element_uuid, initial_props, references={})
    )
    mock_tenant_collection.query.fetch_objects.return_value = MagicMock(objects=[])
    mock_find_element_by_name.return_value = []
    mock_find_relevant_elements.return_value = []
    mock_update_element_properties.return_value = True

    success = await refine_element_description(mock_client, tenant_id, element_uuid)

    assert success is True
    mock_get_element_details.assert_called_once_with(
        mock_client, tenant_id, element_uuid
    )
    expected_updated_props = initial_props.copy()
    expected_updated_props["llm_description"] = "Refined description."
    mock_update_element_properties.assert_called_once()
    args, kwargs = mock_update_element_properties.call_args
    assert args[0] == mock_client
    assert args[1] == tenant_id
    assert args[2] == element_uuid
    assert args[3] == expected_updated_props
    mock_sleep.assert_not_called()


# --- Tests for generate_codebase_summary ---


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@pytest.mark.asyncio
async def test_generate_summary_success(mock_get_provider):
    """Tests successful project summary generation."""
    mock_get_provider.return_value = create_mock_provider(
        "This project contains a main function and a helper class."
    )
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()
    mock_element_collection_mock.tenants = mock_tenants
    mock_element_collection_mock.with_tenant.return_value = mock_tenant_collection
    mock_collections.get.return_value = mock_element_collection_mock
    mock_client.collections = mock_collections
    mock_tenant_collection.query = mock_query

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

    summary = await generate_codebase_summary(mock_client, "SummaryProject")

    assert summary == "This project contains a main function and a helper class."
    mock_tenants.exists.assert_called_once_with("SummaryProject")
    mock_query.fetch_objects.assert_called_once()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@pytest.mark.asyncio
async def test_generate_summary_tenant_not_found(mock_get_provider):
    """Tests summary generation when the tenant doesn't exist."""
    mock_get_provider.return_value = create_mock_provider()
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = False
    mock_collections.get.return_value.tenants = mock_tenants
    mock_client.collections = mock_collections

    summary = await generate_codebase_summary(mock_client, "MissingSummaryProject")

    assert "Error: Codebase 'MissingSummaryProject' not found" in summary
    mock_get_provider.return_value.generate.assert_not_called()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@pytest.mark.asyncio
async def test_generate_summary_no_elements(mock_get_provider):
    """Tests summary generation when no key elements are found."""
    mock_get_provider.return_value = create_mock_provider()
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()
    mock_element_collection_mock.tenants = mock_tenants
    mock_element_collection_mock.with_tenant.return_value = mock_tenant_collection
    mock_collections.get.return_value = mock_element_collection_mock
    mock_client.collections = mock_collections
    mock_tenant_collection.query = mock_query
    mock_query.fetch_objects.return_value = MagicMock(objects=[])

    summary = await generate_codebase_summary(mock_client, "EmptySummaryProject")

    assert "no key functions or classes found" in summary
    mock_query.fetch_objects.assert_called_once()
    mock_get_provider.return_value.generate.assert_not_called()


@patch("src.code_analysis_mcp.llm.get_llm_provider")
@pytest.mark.asyncio
async def test_generate_summary_llm_error(mock_get_provider):
    """Tests summary generation when the LLM call fails."""
    from src.code_analysis_mcp.llm import LLMProviderError

    mock_provider = create_mock_provider()
    mock_provider.generate.side_effect = LLMProviderError("LLM Down")
    mock_get_provider.return_value = mock_provider
    mock_client = MagicMock()
    mock_client.is_connected.return_value = True
    mock_collections = MagicMock()
    mock_tenant_collection = MagicMock()
    mock_query = MagicMock()
    mock_tenants = MagicMock()
    mock_tenants.exists.return_value = True
    mock_element_collection_mock = MagicMock()
    mock_element_collection_mock.tenants = mock_tenants
    mock_element_collection_mock.with_tenant.return_value = mock_tenant_collection
    mock_collections.get.return_value = mock_element_collection_mock
    mock_client.collections = mock_collections
    mock_tenant_collection.query = mock_query
    mock_func = create_mock_weaviate_object(
        "uuid-fn", {"name": "main_func", "element_type": "function"}
    )
    mock_query.fetch_objects.return_value = MagicMock(objects=[mock_func])

    summary = await generate_codebase_summary(mock_client, "LLMSummaryFail")

    assert "Error generating summary" in summary
    assert "LLM Down" in summary
    mock_query.fetch_objects.assert_called_once()
    mock_get_provider.return_value.generate.assert_called_once()
