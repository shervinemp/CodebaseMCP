"""Background task functions: enrichment, summary, topological sort."""

import asyncio
import logging

import weaviate.classes.query as wvc_query

from .code_scanner import enrich_element
from .rag import refine_element_description, generate_codebase_summary
from .weaviate_client import update_codebase_registry

logger = logging.getLogger(__name__)


async def topological_sort_uuids(client, tenant_id: str, uuids: list[str]) -> list[str]:
    """
    Returns UUIDs sorted so callees are enriched before their callers.
    Non-function elements (imports, variables, calls) go first,
    then leaf functions, then their callers, up the call chain.
    """
    collection = client.collections.get("CodeElement")
    response = await asyncio.to_thread(
        collection.with_tenant(tenant_id).query.fetch_objects,
        filters=wvc_query.Filter.by_property("element_type").contains_any(
            ["function", "class"]
        ),
        limit=10000,
        return_references=[
            wvc_query.QueryReference(
                link_on="calls_function",
                return_properties=["name"],
            )
        ],
        return_properties=["name", "element_type"],
    )

    graph: dict[str, list[str]] = {}
    func_uuids: set[str] = set()
    for obj in response.objects:
        uid = str(obj.uuid)
        func_uuids.add(uid)
        graph.setdefault(uid, [])
        refs = obj.references or {}
        if "calls_function" in refs:
            for ref in refs["calls_function"].objects:
                graph[uid].append(str(ref.uuid))

    in_degree = {u: 0 for u in func_uuids}
    for caller, callees in graph.items():
        for callee in callees:
            if callee in in_degree:
                in_degree[callee] += 1

    queue = [u for u in func_uuids if in_degree.get(u, 0) == 0]
    sorted_funcs: list[str] = []
    visited: set[str] = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        sorted_funcs.append(node)
        for callee in graph.get(node, []):
            if callee in in_degree:
                in_degree[callee] -= 1
                if in_degree[callee] == 0:
                    queue.append(callee)

    remaining = [u for u in func_uuids if u not in sorted_funcs]
    sorted_funcs.extend(remaining)

    uuid_set = set(uuids)
    non_funcs = [u for u in uuids if u not in func_uuids]
    result = non_funcs + [u for u in sorted_funcs if u in uuid_set]
    logger.info(
        f"Topological sort: {len(non_funcs)} non-function + {len(sorted_funcs)} function/class elements"
    )
    return result


async def process_element_llm(
    client, uuid: str, tenant_id: str, semaphore: asyncio.Semaphore | None = None
):
    """Enrich and refine a single element."""
    if semaphore is None:
        semaphore = asyncio.Semaphore(5)

    async with semaphore:
        logger.debug(f"Starting LLM processing for {uuid} in tenant {tenant_id}")
        try:
            if not client or not client.is_connected():
                logger.error(
                    f"LLM Task (Tenant: {tenant_id}): Client disconnected for {uuid}. Aborting."
                )
                return
            enriched = await enrich_element(client, tenant_id, uuid)
            if enriched:
                await refine_element_description(client, tenant_id, uuid)
            logger.debug(f"Finished LLM processing for {uuid} in tenant {tenant_id}")
        except Exception as e:
            logger.error(
                f"Error during background LLM processing for {uuid} in tenant {tenant_id}: {e}"
            )


async def background_generate_summary(client, codebase_name: str):
    """Generate and update codebase summary."""
    logger.info(
        f"Background task started: Generating summary for codebase '{codebase_name}'"
    )
    summary = await generate_codebase_summary(client, codebase_name)
    if not summary.startswith("Error:"):
        update_success = update_codebase_registry(
            client, codebase_name, {"summary": summary, "status": "Ready"}
        )
        if update_success:
            logger.info(
                f"Successfully updated summary for codebase '{codebase_name}'. Status set to Ready."
            )
        else:
            logger.error(
                f"Failed to update summary in registry for codebase '{codebase_name}'."
            )
    else:
        logger.error(
            f"Failed to generate summary for codebase '{codebase_name}': {summary}"
        )
        update_codebase_registry(client, codebase_name, {"status": "Error"})
        logger.info(
            f"Set status to Error for codebase '{codebase_name}' due to summary failure."
        )
