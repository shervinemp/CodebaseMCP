"""Full end-to-end test: scan CodebaseMCP, enrich, ask questions."""

# ruff: noqa: E402
import sys
import asyncio

sys.path.insert(0, r"C:\Users\sherv\Desktop\Projects\CodebaseMCP")
from dotenv import load_dotenv

load_dotenv()

from weaviate.classes.tenants import Tenant
from src.code_analysis_mcp.llm import init_llm_provider, get_llm_provider
from src.code_analysis_mcp.weaviate_client import (
    create_weaviate_client,
    create_schema,
    add_codebase_registry_entry,
    update_codebase_registry,
)
from src.code_analysis_mcp.code_scanner import _scan_cleanup_and_upload
from src.code_analysis_mcp.rag import (
    answer_codebase_question,
    generate_codebase_summary,
)
from src.code_analysis_mcp.mcp_server import process_element_llm

TENANT = "codebasemcp"
ROOT = r"C:\Users\sherv\Desktop\Projects\CodebaseMCP"


async def main():
    init_llm_provider()
    p = get_llm_provider()
    print(f"LLM: {p.name} available={p.is_available}")

    client = create_weaviate_client()
    client.connect()
    create_schema(client)

    # Create tenant + registry entry
    for col in ["CodeElement", "CodeFile"]:
        c = client.collections.get(col)
        if not c.tenants.exists(TENANT):
            c.tenants.create([Tenant(name=TENANT)])
    add_codebase_registry_entry(client, TENANT, ROOT, "Scanning")
    print("Tenant + registry created")

    # Scan
    print(f"\nScanning {ROOT} ...")
    status, uuids = await _scan_cleanup_and_upload(client, ROOT, tenant_id=TENANT)
    if "ERROR" in status:
        print(f"  ERROR: {status}")
        return
    print(f"  OK — {len(uuids)} elements")

    # Enrich first 10
    print(f"\nEnriching {min(len(uuids), 10)} elements ...")
    for uuid in uuids[:10]:
        await process_element_llm(client, uuid, TENANT)

    # Summary
    print("\nSummary:")
    summary = await generate_codebase_summary(client, TENANT)
    print(f"  {summary[:400]}")

    # RAG
    questions = [
        "What does create_weaviate_client do?",
        "How is the LLM provider configured?",
        "What MCP tools are available?",
    ]
    print("\nRAG Q&A:")
    for q in questions:
        ans = await answer_codebase_question(q, client=client, tenant_id=TENANT)
        print(f"\n  Q: {q}")
        print(f"  A: {ans[:300]}")

    update_codebase_registry(client, TENANT, {"status": "Ready"})
    client.close()
    print("\nDone!")


asyncio.run(main())
