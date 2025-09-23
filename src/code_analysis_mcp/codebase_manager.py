import asyncio
import os
import logging
import threading
import hashlib
from typing import Any, Dict, Optional, List, Tuple

import weaviate
from weaviate.exceptions import UnexpectedStatusCodeError

from .file_watcher import start_watcher, stop_watcher
from .weaviate_client import (
    create_weaviate_client,
    create_schema,
    get_codebase_details,
    add_codebase_registry_entry,
    update_codebase_registry,
    delete_codebase_registry_entry,
    delete_tenant,
    get_all_codebases,
)
from .code_scanner import _scan_cleanup_and_upload
from .rag import generate_codebase_summary
from .utils import _process_element_properties, _extract_identifiers
from . import config

logger = logging.getLogger(__name__)

class CodebaseManager:
    """Manages codebases, including scanning, state, and file watching."""

    def __init__(self):
        self.client: Optional[weaviate.Client] = None
        self.active_codebase_name: Optional[str] = None
        self.active_watchers: Dict[str, Dict[str, Any]] = {}
        self.background_tasks = set()
        self.llm_semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)
        self.llm_cache: Dict[str, str] = {}


    async def initialize(self):
        """Initializes the Weaviate client and ensures the schema exists."""
        logger.info("CodebaseManager: Initializing...")
        if not self.client:
            self.client = create_weaviate_client()

        if self.client:
            try:
                logger.info("CodebaseManager: Connecting to Weaviate...")
                # The python client no longer requires explicit connect()
                # self.client.connect()

                is_ready = False
                for attempt in range(5):
                    if self.client.is_ready():
                        is_ready = True
                        logger.info("CodebaseManager: Weaviate client is ready.")
                        break
                    logger.warning(f"CodebaseManager: Weaviate not ready (attempt {attempt+1}/5). Waiting...")
                    await asyncio.sleep(2)

                if is_ready:
                    logger.info("CodebaseManager: Ensuring schema exists...")
                    await asyncio.to_thread(create_schema, self.client)
                    logger.info("CodebaseManager: Initialization complete.")
                else:
                    logger.error("CodebaseManager: Weaviate did not become ready.")
                    self.client = None

            except Exception as e:
                logger.exception(f"CodebaseManager: Error during initialization: {e}")
                self.client = None
        else:
            logger.error("CodebaseManager: Failed to create Weaviate client.")

    def shutdown(self):
        """Shuts down the manager, stopping watchers and cleaning up resources."""
        logger.info("CodebaseManager: Shutting down...")

        active_watcher_names = list(self.active_watchers.keys())
        for codebase_name in active_watcher_names:
            self.stop_watcher_sync(codebase_name)

        # Cancel background tasks
        tasks_to_cancel = list(self.background_tasks)
        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} background tasks...")
            for task in tasks_to_cancel:
                task.cancel()

        if self.client and self.client.is_connected():
            logger.info("CodebaseManager: Closing Weaviate client connection.")
            # self.client.close() # New client version may not have close
        logger.info("CodebaseManager: Shutdown complete.")

    async def scan_codebase(self, codebase_name: str, directory: str) -> Dict[str, Any]:
        """Handles codebase scanning, registry updates, tenant creation, analysis, and summary generation."""
        logger.info(f"Scanning codebase '{codebase_name}' at '{directory}'")
        if not self.client or not self.client.is_connected():
            return {"status": "error", "message": "Weaviate client not connected."}

        abs_directory_path = os.path.normpath(directory)
        if not os.path.isdir(abs_directory_path):
            return {"status": "error", "message": f"Directory not found: {abs_directory_path}"}

        try:
            # Check for existing codebase
            if get_codebase_details(self.client, codebase_name):
                return {"status": "error", "message": f"Codebase '{codebase_name}' already exists."}

            # Create tenant
            tenant_id = codebase_name
            for collection_name in ["CodeElement", "CodeFile"]:
                collection = self.client.collections.get(collection_name)
                if not collection.tenants.exists(tenant_id):
                    collection.tenants.create([weaviate.classes.tenants.Tenant(name=tenant_id)])

            # Add to registry
            add_codebase_registry_entry(self.client, codebase_name, abs_directory_path, "Scanning")

            # Scan and upload
            status_message, processed_uuids = await _scan_cleanup_and_upload(self.client, abs_directory_path, tenant_id)
            if "ERROR" in status_message:
                update_codebase_registry(self.client, codebase_name, {"status": "Error"})
                return {"status": "error", "message": f"Scan failed: {status_message}"}

            update_codebase_registry(self.client, codebase_name, {"status": "Summarizing"})
            self.active_codebase_name = codebase_name

            # Start summary generation in background
            summary_task = asyncio.create_task(self.background_generate_summary(codebase_name))
            self.background_tasks.add(summary_task)
            summary_task.add_done_callback(self.background_tasks.discard)

            final_message = f"Scan successful for '{codebase_name}'. Summary generation started. Active codebase set."

            # Start LLM processing in background
            if config.LLM_ENABLED and processed_uuids:
                logger.info(f"Queueing {len(processed_uuids)} elements for LLM processing.")
                for uuid in processed_uuids:
                    llm_task = asyncio.create_task(self._process_element_llm(uuid, tenant_id))
                    self.background_tasks.add(llm_task)
                    llm_task.add_done_callback(self.background_tasks.discard)
                final_message += f" Background LLM enrichment started for {len(processed_uuids)} elements."

            # Start watcher
            watcher_started, watcher_msg = self.start_watcher(codebase_name)
            final_message += f" {watcher_msg}"

            return {"status": "success", "message": final_message, "codebase_name": codebase_name}

        except Exception as e:
            logger.exception(f"Error during scan_codebase for '{codebase_name}': {e}")
            update_codebase_registry(self.client, codebase_name, {"status": "Error"})
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

    async def _process_element_llm(self, uuid: str, tenant_id: str):
        """Enriches and then refines a single element using LLM calls, with caching."""
        from .rag import refine_element_description
        from .code_scanner import enrich_element
        from .weaviate_client import get_element_details, update_element_properties

        async with self.llm_semaphore:
            element_details = get_element_details(self.client, tenant_id, uuid)
            if not element_details or not element_details.properties.get("code_snippet"):
                return

            code_snippet = element_details.properties["code_snippet"]
            cache_key = hashlib.sha256(code_snippet.encode()).hexdigest()

            if cache_key in self.llm_cache:
                description = self.llm_cache[cache_key]
                update_element_properties(self.client, tenant_id, uuid, {"llm_description": description})
                logger.info(f"LLM Cache HIT for element {uuid}")
                return

            logger.info(f"LLM Cache MISS for element {uuid}. Processing...")
            try:
                enriched = await enrich_element(self.client, tenant_id, uuid)
                if enriched:
                    description = await refine_element_description(self.client, tenant_id, uuid)
                    if description:
                        self.llm_cache[cache_key] = description
                logger.debug(f"Finished LLM processing for {uuid} in tenant {tenant_id}")
            except Exception as e:
                logger.error(f"Error during background LLM processing for {uuid}: {e}")

    async def background_generate_summary(self, codebase_name: str):
        """Background task to generate and update codebase summary."""
        logger.info(f"Generating summary for codebase '{codebase_name}'")
        summary = await generate_codebase_summary(self.client, codebase_name)
        if not summary.startswith("Error:"):
            update_codebase_registry(self.client, codebase_name, {"summary": summary, "status": "Ready"})
            logger.info(f"Successfully updated summary for '{codebase_name}'.")
        else:
            update_codebase_registry(self.client, codebase_name, {"status": "Error"})
            logger.error(f"Failed to generate summary for '{codebase_name}': {summary}")

    def list_codebases(self) -> Dict[str, Any]:
        if not self.client or not self.client.is_connected():
            return {"status": "error", "message": "Weaviate client not connected."}
        try:
            codebases = get_all_codebases(self.client)
            return {"status": "success", "codebases": codebases}
        except Exception as e:
            logger.exception("Error listing codebases")
            return {"status": "error", "message": str(e)}

    def select_codebase(self, codebase_name: str) -> Dict[str, Any]:
        if not self.client or not self.client.is_connected():
            return {"status": "error", "message": "Weaviate client not connected."}

        details = get_codebase_details(self.client, codebase_name)
        if not details:
            return {"status": "error", "message": f"Codebase '{codebase_name}' not found."}

        if self.active_codebase_name and self.active_codebase_name != codebase_name:
            self.stop_watcher_sync(self.active_codebase_name)

        self.active_codebase_name = codebase_name
        summary = details.get("summary", "No summary available.")
        return {"status": "success", "message": f"Codebase '{codebase_name}' selected. Summary: {summary}"}

    def delete_codebase(self, codebase_name: str) -> Dict[str, Any]:
        if not self.client or not self.client.is_connected():
            return {"status": "error", "message": "Weaviate client not connected."}

        self.stop_watcher_sync(codebase_name)

        tenant_deleted = delete_tenant(self.client, codebase_name)
        registry_deleted = delete_codebase_registry_entry(self.client, codebase_name)

        if self.active_codebase_name == codebase_name:
            self.active_codebase_name = None

        if not tenant_deleted or not registry_deleted:
            return {"status": "error", "message": "Failed to completely delete codebase. Check logs."}

        return {"status": "success", "message": f"Codebase '{codebase_name}' deleted."}

    def start_watcher(self, codebase_name: str) -> Tuple[bool, str]:
        """Starts the file watcher for a given codebase."""
        return start_watcher(self, self.active_watchers, codebase_name)

    def stop_watcher_sync(self, codebase_name: str) -> Tuple[bool, str]:
        """Stops the file watcher synchronously."""
        return stop_watcher(self, self.active_watchers, codebase_name)

    async def handle_file_deletion(self, codebase_name: str, file_path: str):
        """Handles the deletion of a file from the codebase."""
        logger.info(f"Handling file deletion for {file_path} in {codebase_name}")
        from .weaviate_client import delete_elements_by_file_path, delete_code_file
        await asyncio.to_thread(delete_elements_by_file_path, self.client, codebase_name, file_path)
        await asyncio.to_thread(delete_code_file, self.client, codebase_name, file_path)

    async def handle_rescan(self, codebase_name: str):
        """Handles a rescan of the codebase directory."""
        logger.info(f"Handling rescan for {codebase_name}")
        details = get_codebase_details(self.client, codebase_name)
        if not details or not details.get("directory"):
            logger.error(f"Could not find directory for codebase {codebase_name} to rescan.")
            return

        directory = details["directory"]
        status_message, processed_uuids = await _scan_cleanup_and_upload(self.client, directory, codebase_name)

        if "ERROR" in status_message:
            logger.error(f"Rescan failed for {codebase_name}: {status_message}")
            return

        if config.LLM_ENABLED and processed_uuids:
            logger.info(f"Queueing {len(processed_uuids)} elements for LLM processing after rescan.")
            for uuid in processed_uuids:
                llm_task = asyncio.create_task(self._process_element_llm(uuid, codebase_name))
                self.background_tasks.add(llm_task)
                llm_task.add_done_callback(self.background_tasks.discard)

    async def find_element(self, name: Optional[str] = None, file_path: Optional[str] = None, element_type: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        if not self.client or not self.active_codebase_name:
            return {"status": "error", "message": "Select a codebase first."}

        from .weaviate_client import find_element_by_name

        tenant_id = self.active_codebase_name
        results = find_element_by_name(self.client, [tenant_id], name, file_path, element_type, limit)

        codebase_details = get_codebase_details(self.client, tenant_id)
        codebase_root = codebase_details.get("directory") if codebase_details else None

        processed_results = [
            await _process_element_properties(
                self.client, tenant_id, r.properties, str(r.uuid), "list", codebase_root
            ) for r in results
        ]
        return {"status": "success", "count": len(processed_results), "elements": processed_results}

    async def get_details(self, uuid: str) -> Dict[str, Any]:
        if not self.client or not self.active_codebase_name:
            return {"status": "error", "message": "Select a codebase first."}

        from .weaviate_client import get_element_details

        tenant_id = self.active_codebase_name
        details = get_element_details(self.client, tenant_id, uuid)
        if not details:
            return {"status": "not_found", "message": f"Element '{uuid}' not found."}

        codebase_details = get_codebase_details(self.client, tenant_id)
        codebase_root = codebase_details.get("directory") if codebase_details else None

        processed_details = await _process_element_properties(
            self.client, tenant_id, details.properties, uuid, "detail", codebase_root
        )
        return {"status": "success", "details": processed_details}

    async def ask_question(self, query: str) -> Dict[str, Any]:
        if not self.client or not self.active_codebase_name:
            return {"status": "error", "message": "Select a codebase first."}

        from .rag import answer_codebase_question

        answer = await answer_codebase_question(query, self.client, self.active_codebase_name)
        if answer.startswith("ERROR:"):
            return {"status": "error", "message": answer}
        return {"status": "success", "answer": answer}

    def queue_llm_processing(self, codebase_name: str, uuids: List[str], skip_enriched: bool = True) -> bool:
        """Placeholder for queuing LLM processing tasks."""
        logger.info(f"Placeholder: Queued LLM processing for {len(uuids)} items in {codebase_name}.")
        # In a real implementation, this would add tasks to a queue.
        return True

    async def analyze_snippet(self, code_snippet: str) -> Dict[str, Any]:
        if not self.client or not self.active_codebase_name:
            return {"status": "error", "message": "Select a codebase first."}

        identifiers = _extract_identifiers(code_snippet)
        if not identifiers:
            return {"status": "success", "message": "No identifiers found.", "related_elements": []}

        unique_elements = {}
        for ident in identifiers:
            result = await self.find_element(name=ident, limit=3)
            if result.get("status") == "success":
                for element in result.get("elements", []):
                    unique_elements[element["uuid"]] = element

        return {"status": "success", "related_elements": list(unique_elements.values())}
