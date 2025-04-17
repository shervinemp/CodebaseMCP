import logging
import asyncio
from .weaviate_client import (  # Use relative import
    create_weaviate_client,
)
import weaviate.classes.query as wvc_query

logger = logging.getLogger(__name__)


async def generate_mermaid_call_graph(client=None, start_node_uuid=None) -> str:
    """
    Queries Weaviate for function/method calls and generates a MermaidJS graph.
    If start_node_uuid is provided, attempts to generate a graph centered around that node.
    Otherwise, generates a graph of all known call relationships.
    """
    logger.info(f"Generating Mermaid call graph. Start node UUID: {start_node_uuid}")
    client_created_internally = False
    weaviate_client = client

    mermaid_lines = ["graph TD"]
    nodes = set()
    edges = set()

    try:
        if weaviate_client is None:
            logger.debug("generate_mermaid_call_graph: Creating internal client...")
            weaviate_client = create_weaviate_client()
            if not weaviate_client:
                return "graph TD\n  Error[Error generating graph: Could not create Weaviate client]"
            client_created_internally = True
            weaviate_client.connect()
            if not weaviate_client.is_connected():
                logger.error(
                    "generate_mermaid_call_graph: Failed to connect internal Weaviate client."
                )
                return "graph TD\n  Error[Error generating graph: Could not connect internal Weaviate client]"

        # If a client was passed in, ensure it's connected
        if not client_created_internally and not weaviate_client.is_connected():
            logger.debug(
                "generate_mermaid_call_graph: Passed-in client not connected, connecting..."
            )
            weaviate_client.connect()
            if not weaviate_client.is_connected():
                logger.error(
                    "generate_mermaid_call_graph: Failed to connect passed-in Weaviate client."
                )
                return "graph TD\n  Error[Error generating graph: Could not connect passed-in Weaviate client]"

        elements_collection = weaviate_client.collections.get("CodeElement")

        # --- Fetch relevant elements ---
        logger.debug("Fetching functions and methods...")
        response_funcs = await asyncio.to_thread(
            elements_collection.query.fetch_objects,
            filters=wvc_query.Filter.by_property("element_type").equal("function"),
            limit=1000,
            return_properties=["name", "file_path"],
            return_references=[
                wvc_query.QueryReference(
                    link_on="calls_function",
                    return_properties=["name", "element_type"],
                )
            ],
        )

        logger.debug(f"Fetched {len(response_funcs.objects)} functions/methods.")

        # --- Process elements and build graph components ---
        for func_obj in response_funcs.objects:
            caller_uuid = str(func_obj.uuid)
            caller_name = func_obj.properties.get("name", "UnknownFunc")
            caller_file = func_obj.properties.get("file_path", "UnknownFile")
            caller_label = f"{caller_name} ({caller_file.split('/')[-1]})"
            caller_node_id = f"N_{caller_uuid.replace('-', '')}"

            if caller_node_id not in nodes:
                nodes.add(caller_node_id)
                mermaid_lines.append(f'  {caller_node_id}["{caller_label}"]')

            # Process references (calls)
            calls_ref = (
                func_obj.references.get("calls_function")
                if func_obj.references
                else None
            )
            if calls_ref:
                for called_obj_ref in calls_ref.objects:
                    callee_uuid = str(called_obj_ref.uuid)
                    callee_name = called_obj_ref.properties.get(
                        "name", "UnknownCalledFunc"
                    )
                    callee_label = f"{callee_name}"
                    callee_node_id = f"N_{callee_uuid.replace('-', '')}"

                    if callee_node_id not in nodes:
                        nodes.add(callee_node_id)
                        mermaid_lines.append(f'  {callee_node_id}["{callee_label}"]')

                    edge = (caller_node_id, callee_node_id)
                    if edge not in edges:
                        edges.add(edge)
                        mermaid_lines.append(f"  {caller_node_id} --> {callee_node_id}")

        if not edges:
            mermaid_lines.append("  NoCallsDetected((No call relationships found))")

        logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges.")
        return "\n".join(mermaid_lines)

    except Exception as e:
        logger.exception(f"Error generating Mermaid graph: {e}")
        return f"graph TD\n  Error[Error generating graph: {e}]"
    finally:
        if (
            client_created_internally
            and weaviate_client
            and weaviate_client.is_connected()
        ):
            logger.debug("generate_mermaid_call_graph: Closing internal client.")
            weaviate_client.close()
