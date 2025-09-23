import logging
import asyncio
from .weaviate_client import (  # Use relative import
    create_weaviate_client,
)
import weaviate.classes.query as wvc_query

logger = logging.getLogger(__name__)


def _add_node(uuid: str, properties: dict, nodes: set, mermaid_lines: list):
    """Helper to add a node to the graph if it doesn't exist."""
    node_id = f"N_{uuid.replace('-', '')}"
    if node_id in nodes:
        return node_id

    name = properties.get("name", "Unknown")
    file = properties.get("file_path", "UnknownFile")
    label = f"{name} ({file.split('/')[-1]})"

    nodes.add(node_id)
    mermaid_lines.append(f'  {node_id}["{label}"]')
    return node_id

def _add_edge(caller_id: str, callee_id: str, edges: set, mermaid_lines: list):
    """Helper to add an edge to the graph if it doesn't exist."""
    edge = (caller_id, callee_id)
    if edge in edges:
        return
    edges.add(edge)
    mermaid_lines.append(f"  {caller_id} --> {callee_id}")


async def generate_mermaid_call_graph(client=None, start_node_uuid=None) -> str:
    """
    Queries Weaviate for function/method calls and generates a MermaidJS graph.
    If start_node_uuid is provided, generates a graph centered around that node.
    Otherwise, generates a graph of all known call relationships.
    """
    logger.info(f"Generating Mermaid call graph. Start node UUID: {start_node_uuid}")
    weaviate_client = client
    # This function assumes the client is passed in and connected.
    # The mcp_server or CodebaseManager should handle client lifecycle.
    if not weaviate_client or not weaviate_client.is_connected():
        return "graph TD\n  Error[Weaviate client not provided or not connected]"

    mermaid_lines = ["graph TD"]
    nodes = set()
    edges = set()

    try:
        elements_collection = weaviate_client.collections.get("CodeElement")

        if start_node_uuid:
            # Focused graph logic
            # 1. Get the start node
            start_node_obj = await asyncio.to_thread(
                elements_collection.query.fetch_object_by_id,
                start_node_uuid,
                return_properties=["name", "file_path"],
                return_references=[wvc_query.QueryReference(link_on="calls_function")]
            )
            if not start_node_obj:
                return f"graph TD\n  Error[Node with UUID {start_node_uuid} not found]"

            start_node_id = _add_node(start_node_uuid, start_node_obj.properties, nodes, mermaid_lines)

            # 2. Get callees (functions called by the start node)
            if start_node_obj.references and "calls_function" in start_node_obj.references:
                for callee_ref in start_node_obj.references["calls_function"].objects:
                    callee_id = _add_node(str(callee_ref.uuid), callee_ref.properties, nodes, mermaid_lines)
                    _add_edge(start_node_id, callee_id, edges, mermaid_lines)

            # 3. Get callers (functions that call the start node)
            callers_response = await asyncio.to_thread(
                elements_collection.query.fetch_objects,
                filters=wvc_query.Filter.by_reference("calls_function").contains_any([start_node_uuid]),
                limit=50, # Limit callers to keep graph reasonable
                return_properties=["name", "file_path"]
            )
            for caller_obj in callers_response.objects:
                caller_id = _add_node(str(caller_obj.uuid), caller_obj.properties, nodes, mermaid_lines)
                _add_edge(caller_id, start_node_id, edges, mermaid_lines)

        else:
            # Full graph logic (existing behavior)
            response_funcs = await asyncio.to_thread(
                elements_collection.query.fetch_objects,
                filters=wvc_query.Filter.by_property("element_type").contains_any(["function", "method"]),
                limit=1000,
                return_properties=["name", "file_path"],
                return_references=[wvc_query.QueryReference(link_on="calls_function")]
            )

            for func_obj in response_funcs.objects:
                caller_id = _add_node(str(func_obj.uuid), func_obj.properties, nodes, mermaid_lines)
                if func_obj.references and "calls_function" in func_obj.references:
                    for called_obj_ref in func_obj.references["calls_function"].objects:
                        callee_id = _add_node(str(called_obj_ref.uuid), called_obj_ref.properties, nodes, mermaid_lines)
                        _add_edge(caller_id, callee_id, edges, mermaid_lines)

        if not edges:
            mermaid_lines.append("  NoCallsDetected((No call relationships found))")

        logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges.")
        return "\n".join(mermaid_lines)

    except Exception as e:
        logger.exception(f"Error generating Mermaid graph: {e}")
        return f"graph TD\n  Error[Error generating graph: {e}]"
