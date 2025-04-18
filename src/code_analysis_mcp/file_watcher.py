import os
import logging
import asyncio
import time
import threading
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileModifiedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
)
from typing import Dict, Any, List, Tuple

# Assuming these imports are needed based on the moved code
from .weaviate_client import WeaviateManager
from .code_scanner import (
    CodeScanner,
    GENERATE_LLM_DESCRIPTIONS,
)  # Import config flag

logger = logging.getLogger(__name__)

# --- Configuration (Consider moving to a central config if shared) ---
WATCHER_POLLING_INTERVAL = int(os.getenv("WATCHER_POLLING_INTERVAL", "5"))
RESCAN_DEBOUNCE_DELAY = float(os.getenv("RESCAN_DEBOUNCE_DELAY", "5.0"))  # Seconds


# --- File Watcher Logic ---


class AnalysisTriggerHandler(FileSystemEventHandler):
    """Triggers analysis when a watched file is modified or created."""

    def __init__(
        self, manager: WeaviateManager, codebase_name: str, patterns: list[str]
    ):
        self.manager = manager  # Store the manager instance
        self.codebase_name = codebase_name
        self.patterns = patterns
        self.last_event_time: Dict[str, float] = {}
        self._rescan_timer: asyncio.TimerHandle | None = (
            None  # Timer for debouncing rescans
        )
        try:
            # Get the running loop in the thread where the handler is instantiated
            # This loop belongs to the mcp_server's main thread usually
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running (e.g., in the watcher thread before starting its own loop)
            self.loop = None
            logger.warning(
                "AnalysisTriggerHandler initialized without a running asyncio loop."
            )

    def _should_process(self, event_path):
        """Check if the event path matches the patterns."""
        if not event_path:
            return False
        # Basic check, might need refinement for more complex patterns
        return any(event_path.endswith(p.strip("*.")) for p in self.patterns)

    def _run_async_task(self, coro):
        """Safely run an async task from a sync handler thread."""
        if self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                # Increased timeout as scanning can take time
                future.result(timeout=180)
            except TimeoutError:
                logger.error(
                    f"Watcher: Async task timed out for codebase {self.codebase_name}"
                )
            except Exception as e:
                logger.error(
                    f"Watcher: Async task failed for codebase {self.codebase_name}: {e}"
                )
        else:
            # Fallback: Run in a new event loop (less ideal)
            logger.warning(
                f"Watcher: Running async task in new event loop for {self.codebase_name}."
            )
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.error(
                    f"Watcher: Async task failed (new loop) for codebase {self.codebase_name}: {e}"
                )

    def process(self, event):
        """Process file system event: Scan and Upload for the specific codebase."""
        if event.is_directory or not self._should_process(event.src_path):
            return

        event_type = event.event_type
        # Use absolute path for consistency
        path = os.path.abspath(event.src_path)
        debounce_period = 2.0
        current_time = time.time()

        last_time = self.last_event_time.get(path, 0)
        if current_time - last_time < debounce_period:
            logger.debug(f"Debouncing {event_type} event for: {path}")
            return

        self.last_event_time[path] = current_time
        logger.info(
            f"Watcher: Detected {event_type} for {path} in codebase '{self.codebase_name}'. Queueing update check."
        )

        if not self.manager or not self.manager.client:
            logger.error(
                f"Watcher: Weaviate manager/client unavailable for {self.codebase_name}. Aborting event processing."
            )
            return

        try:
            if event_type == "deleted":
                logger.info(
                    f"Watcher: Deleting data for {path} in tenant '{self.codebase_name}'"
                )
                # Use manager methods (run in thread to avoid blocking watchdog)
                asyncio.run_coroutine_threadsafe(
                    self._delete_file_data(path), self.loop or asyncio.new_event_loop()
                )
            # --- Global Rescan Debounce Logic ---
            elif event_type in ["modified", "created"]:
                if not self.loop:
                    logger.error(
                        f"Watcher: No event loop available for {self.codebase_name} to schedule rescan."
                    )
                    return

                # Cancel any pending rescan timer
                if self._rescan_timer:
                    self._rescan_timer.cancel()
                    logger.debug(
                        f"Watcher: Canceled pending rescan timer for {self.codebase_name}"
                    )

                # Schedule the actual rescan trigger after a delay
                logger.info(
                    f"Watcher: Scheduling rescan for {self.codebase_name} in {RESCAN_DEBOUNCE_DELAY}s"
                )
                self._rescan_timer = self.loop.call_later(
                    RESCAN_DEBOUNCE_DELAY, self._schedule_rescan_coroutine
                )
            # --- End Global Rescan Debounce Logic ---

        except Exception as e:
            logger.error(
                f"Watcher: Error processing event for {path} in codebase '{self.codebase_name}': {e}"
            )

    async def _delete_file_data(self, abs_path: str):
        """Async helper to delete file data using the manager."""
        try:
            logger.info(f"Async delete task started for {abs_path}")
            await asyncio.to_thread(
                self.manager.delete_elements_by_file_path, self.codebase_name, abs_path
            )
            await asyncio.to_thread(
                self.manager.delete_code_file, self.codebase_name, abs_path
            )
            logger.info(f"Async delete task finished for {abs_path}")
        except Exception as e:
            logger.error(f"Async delete task failed for {abs_path}: {e}")

    def _schedule_rescan_coroutine(self):
        """Schedules the _rescan_codebase coroutine to run thread-safely."""
        if not self.loop or not self.loop.is_running():
            logger.error(
                f"Watcher: Event loop stopped before debounced rescan could run for {self.codebase_name}."
            )
            return
        logger.info(
            f"Watcher: Debounce period ended for {self.codebase_name}. Triggering rescan coroutine."
        )
        # Ensure the coroutine is run in the loop the handler was initialized with
        asyncio.run_coroutine_threadsafe(self._rescan_codebase(), self.loop)
        self._rescan_timer = None  # Clear the timer handle

    async def _rescan_codebase(self):
        """Async helper to trigger a rescan of the entire codebase directory. Should be run via run_coroutine_threadsafe."""
        # Ensure this runs only once per debounce period by checking timer again (belt and suspenders)
        # This check might be redundant if scheduling logic is perfect, but adds safety.
        # if self._rescan_timer is not None:
        #     logger.warning(f"Watcher: _rescan_codebase called while timer still active for {self.codebase_name}. Skipping redundant run.")
        #     return

        processed_uuids = []
        scan_success = False
        try:
            logger.info(f"Async rescan task started for {self.codebase_name}")
            codebase_details = await asyncio.to_thread(
                self.manager.get_codebase_details, self.codebase_name
            )
            if codebase_details and codebase_details.get("directory"):
                codebase_dir = codebase_details["directory"]
                # Instantiate CodeScanner and call its method
                # Assuming CodeScanner doesn't need the client passed directly anymore if manager handles it
                scanner = CodeScanner(self.manager)
                # Capture the result and UUIDs
                scan_status, processed_uuids = await scanner._scan_cleanup_and_upload(
                    codebase_dir, tenant_id=self.codebase_name
                )
                logger.info(
                    f"Async rescan task finished for {self.codebase_name}. Status: {scan_status}"
                )
                # Check if scan reported success (adjust based on actual return value if needed)
                scan_success = "ERROR" not in scan_status.upper()

            else:
                logger.error(
                    f"Watcher: Could not get codebase directory for '{self.codebase_name}' to trigger scan."
                )
        except Exception as e:
            logger.error(f"Async rescan task failed for {self.codebase_name}: {e}")
            scan_success = False

        # --- Trigger LLM Processing for Updated Elements ---
        if scan_success and processed_uuids and GENERATE_LLM_DESCRIPTIONS:
            logger.info(
                f"Watcher: Queueing {len(processed_uuids)} updated elements for LLM processing in {self.codebase_name}."
            )
            try:
                # This assumes a method exists on the manager to handle queuing.
                # The actual implementation of this method would live elsewhere (e.g., weaviate_client.py or mcp_server.py)
                # and interact with the background processing queue.
                success = await asyncio.to_thread(
                    self.manager.queue_llm_processing,
                    self.codebase_name,
                    processed_uuids,
                    skip_enriched=True,  # Usually skip elements already processed unless forced
                )
                if success:
                    logger.info(
                        f"Watcher: Successfully queued elements for LLM processing in {self.codebase_name}."
                    )
                else:
                    logger.error(
                        f"Watcher: Failed to queue elements for LLM processing in {self.codebase_name} via manager."
                    )
            except AttributeError:
                logger.error(
                    f"Watcher: WeaviateManager does not have the 'queue_llm_processing' method. LLM re-processing trigger skipped."
                )
            except Exception as llm_e:
                logger.error(
                    f"Watcher: Error queueing elements for LLM processing in {self.codebase_name}: {llm_e}"
                )
        elif scan_success and not processed_uuids:
            logger.info(
                f"Watcher: Rescan complete for {self.codebase_name}, but no elements were processed/updated."
            )
        elif not scan_success:
            logger.warning(
                f"Watcher: Rescan failed for {self.codebase_name}, LLM processing trigger skipped."
            )
        # --- End LLM Processing Trigger ---

    def on_modified(self, event: FileModifiedEvent):
        self.process(event)

    def on_created(self, event: FileCreatedEvent):
        self.process(event)

    def on_deleted(self, event: FileDeletedEvent):
        self.process(event)


def watcher_thread_target(
    manager: WeaviateManager,  # Pass manager instance
    active_watchers_dict: Dict[str, Any],  # Pass the dict to check status
    codebase_name: str,
    directory: str,
    stop_event: threading.Event,
):
    """Target function for the watcher thread."""
    logger.info(
        f"Watcher thread started for codebase '{codebase_name}' on directory '{directory}'"
    )
    patterns = ["*.py"]  # Consider making patterns configurable

    if not manager or not manager.client:
        logger.error(
            f"Watcher thread for '{codebase_name}': Weaviate manager not initialized. Stopping."
        )
        return

    # Create a new event loop for this thread if one doesn't exist
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    event_handler = AnalysisTriggerHandler(
        manager=manager,  # Pass the manager
        codebase_name=codebase_name,
        patterns=patterns,
    )
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    logger.info(f"Observer started for '{codebase_name}'.")

    try:
        while not stop_event.wait(timeout=WATCHER_POLLING_INTERVAL):
            # Check if manager and its client are available
            if not manager or not manager.client:
                logger.warning(
                    f"Watcher thread for '{codebase_name}': Weaviate manager/client unavailable. Stopping watcher."
                )
                break

            # Check status directly from the passed-in dictionary (managed by mcp_server)
            # This avoids needing the manager for this check, reducing potential async issues here.
            watcher_info = active_watchers_dict.get(codebase_name)
            if not watcher_info or watcher_info.get("stop_event").is_set():
                logger.info(
                    f"Watcher thread for '{codebase_name}': Stop signal received or watcher removed from active list. Stopping."
                )
                break

            # Optional: Could still check registry status periodically via manager if needed for external consistency
            # details = manager.get_codebase_details(codebase_name) # This would need to be run in thread
            # if not details or not details.get("watcher_active", False):
            #     logger.info(f"Watcher thread for '{codebase_name}': watcher_active flag is False in registry. Stopping watcher.")
            #     stop_event.set() # Signal stop based on registry
            #     break

            logger.debug(
                f"Watcher thread for '{codebase_name}': Still active, polling again."
            )

    except Exception as e:
        logger.error(f"Watcher thread for '{codebase_name}' encountered an error: {e}")
    finally:
        logger.info(f"Watcher thread for '{codebase_name}': Stopping observer...")
        observer.stop()
        observer.join()
        logger.info(
            f"Watcher thread for '{codebase_name}': Observer stopped and joined."
        )
        # Registry update is handled by the stop_watcher function in mcp_server


def start_watcher(
    manager: WeaviateManager, active_watchers_dict: Dict[str, Any], codebase_name: str
) -> Tuple[bool, str]:
    """
    Starts the file watcher for a given codebase in a separate thread.
    Manages the active_watchers_dict provided by the caller.
    """
    logger.info(f"Attempting to start watcher for codebase '{codebase_name}'")

    if not manager or not manager.client:
        msg = "Cannot start watcher: Weaviate manager/client not available."
        logger.error(msg)
        return False, msg

    # Use manager method
    details = manager.get_codebase_details(codebase_name)
    if not details:
        msg = f"Cannot start watcher: Codebase '{codebase_name}' not found in registry."
        logger.error(msg)
        return False, msg

    directory = details.get("directory")
    if not directory or not os.path.isdir(directory):
        msg = f"Cannot start watcher: Codebase '{codebase_name}' directory '{directory}' not found or invalid."
        logger.error(msg)
        return False, msg

    # Check the passed-in dictionary for active watchers
    if codebase_name in active_watchers_dict:
        msg = f"Watcher for codebase '{codebase_name}' is already running in this server instance."
        logger.warning(msg)
        return False, msg  # Prevent duplicate threads locally

    # Check registry status (optional, but good practice)
    if details.get("watcher_active", False):
        msg = f"Watcher for codebase '{codebase_name}' appears to be active in registry (possibly another instance?). Cannot start duplicate."
        logger.warning(msg)
        # Decide if this should be an error or just a warning
        # return False, msg

    try:
        # Update registry status first
        if not manager.update_codebase_registry(
            codebase_name, {"watcher_active": True}
        ):
            msg = f"Failed to update watcher status in registry for codebase '{codebase_name}'."
            logger.error(msg)
            return False, msg

        stop_event = threading.Event()
        thread = threading.Thread(
            target=watcher_thread_target,
            args=(
                manager,
                active_watchers_dict,
                codebase_name,
                directory,
                stop_event,
            ),  # Pass manager and dict
            daemon=True,
        )
        thread.start()

        # Update the dictionary managed by the caller (mcp_server)
        active_watchers_dict[codebase_name] = {
            "thread": thread,
            "stop_event": stop_event,
            "directory": directory,
        }
        msg = f"File watcher started successfully for codebase '{codebase_name}' on directory '{directory}'."
        logger.info(msg)
        return True, msg

    except Exception as e:
        logger.exception(f"Failed to start watcher for codebase '{codebase_name}': {e}")
        # Attempt to revert registry status
        if manager and manager.client:
            manager.update_codebase_registry(codebase_name, {"watcher_active": False})
        # Clean up local dict if entry was added prematurely
        if codebase_name in active_watchers_dict:
            del active_watchers_dict[codebase_name]
        return False, f"Failed to start watcher: {e}"


def stop_watcher(
    manager: WeaviateManager, active_watchers_dict: Dict[str, Any], codebase_name: str
) -> Tuple[bool, str]:
    """
    Signals a watcher thread to stop, updates the registry, and removes from active_watchers_dict.
    """
    logger.info(f"Attempting to stop watcher for codebase '{codebase_name}'")

    if not manager or not manager.client:
        msg = "Cannot stop watcher: Weaviate manager/client not available."
        logger.error(msg)
        return False, msg

    # Update registry first to prevent restarts
    logger.info(f"Setting watcher_active=False in registry for '{codebase_name}'")
    update_success = manager.update_codebase_registry(
        codebase_name, {"watcher_active": False}
    )
    if not update_success:
        logger.error(
            f"Failed to update watcher status in registry for '{codebase_name}', but attempting local stop."
        )
        # Continue with local stop attempt despite registry failure

    # Check the passed-in dictionary
    watcher_info = active_watchers_dict.get(codebase_name)
    if watcher_info:
        logger.info(
            f"Found active watcher for '{codebase_name}' in this instance. Signaling stop."
        )
        stop_event = watcher_info.get("stop_event")
        thread = watcher_info.get("thread")

        if stop_event:
            stop_event.set()  # Signal the thread to stop

        # Remove from the active dictionary immediately
        if codebase_name in active_watchers_dict:
            del active_watchers_dict[codebase_name]
            logger.info(f"Removed '{codebase_name}' from active watcher dictionary.")

        if thread and thread.is_alive():
            logger.info(
                f"Waiting briefly for watcher thread '{codebase_name}' to join..."
            )
            thread.join(timeout=WATCHER_POLLING_INTERVAL + 2)  # Wait for thread to exit
            if thread.is_alive():
                logger.warning(
                    f"Watcher thread '{codebase_name}' did not exit cleanly after stop signal."
                )

        msg = f"Stop signal sent to local watcher for codebase '{codebase_name}'. Registry updated. Removed from active list."
        logger.info(msg)
        return True, msg
    else:
        # If not found locally, ensure registry is inactive and report success
        msg = f"No active watcher found for codebase '{codebase_name}' in this server instance. Registry status set to inactive."
        logger.info(msg)
        return True, msg  # Return true because the desired state (inactive) is achieved
