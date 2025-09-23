import os
import logging
import asyncio
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, Any, Tuple
import weaviate.classes.query as wvc_query

from .codebase_manager import CodebaseManager
from . import config

logger = logging.getLogger(__name__)

class AnalysisTriggerHandler(FileSystemEventHandler):
    """Triggers analysis when a watched file is modified, created, or deleted."""

    def __init__(self, manager: 'CodebaseManager', codebase_name: str):
        self.manager = manager
        self.codebase_name = codebase_name
        self.patterns = config.WATCHER_PATTERNS
        self._rescan_timer: Optional[asyncio.TimerHandle] = None

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _should_process(self, event_path):
        return any(event_path.endswith(p.strip("*")) for p in self.patterns)

    def process(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return

        if event.event_type == "deleted":
            asyncio.run_coroutine_threadsafe(
                self.manager.handle_file_deletion(self.codebase_name, event.src_path), self.loop
            )
        else: # modified or created
            if self._rescan_timer:
                self._rescan_timer.cancel()

            self._rescan_timer = self.loop.call_later(
                config.RESCAN_DEBOUNCE_DELAY,
                lambda: asyncio.run_coroutine_threadsafe(self.manager.handle_rescan(self.codebase_name), self.loop)
            )

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)

def watcher_thread_target(manager: 'CodebaseManager', codebase_name: str, directory: str, stop_event: threading.Event):
    """Target function for the watcher thread."""
    observer = Observer()
    handler = AnalysisTriggerHandler(manager, codebase_name)
    observer.schedule(handler, directory, recursive=True)
    observer.start()
    stop_event.wait()
    observer.stop()
    observer.join()

def start_watcher(manager: 'CodebaseManager', active_watchers_dict: Dict[str, Any], codebase_name: str) -> Tuple[bool, str]:
    if codebase_name in active_watchers_dict:
        return False, f"Watcher for '{codebase_name}' is already running."

    if not manager.client:
        return False, "Weaviate client not available."

    details = manager.client.collections.get("CodebaseRegistry").query.fetch_objects(
        filters=wvc_query.Filter.by_property("codebase_name").equal(codebase_name),
        limit=1
    )

    if not details.objects:
        return False, f"Codebase '{codebase_name}' not found."

    directory = details.objects[0].properties.get("directory")
    if not directory or not os.path.isdir(directory):
        return False, f"Directory '{directory}' not found or invalid."

    stop_event = threading.Event()
    thread = threading.Thread(
        target=watcher_thread_target,
        args=(manager, codebase_name, directory, stop_event),
        daemon=True,
    )
    thread.start()

    active_watchers_dict[codebase_name] = {
        "thread": thread,
        "stop_event": stop_event,
        "directory": directory,
    }
    return True, f"Watcher started for '{codebase_name}'."

def stop_watcher(manager: 'CodebaseManager', active_watchers_dict: Dict[str, Any], codebase_name: str) -> Tuple[bool, str]:
    watcher_info = active_watchers_dict.get(codebase_name)
    if not watcher_info:
        return False, f"No active watcher found for '{codebase_name}'."

    watcher_info["stop_event"].set()
    watcher_info["thread"].join(timeout=5)

    if watcher_info["thread"].is_alive():
        logger.warning(f"Watcher thread for '{codebase_name}' did not stop gracefully.")

    del active_watchers_dict[codebase_name]

    if manager.client:
        details = manager.client.collections.get("CodebaseRegistry").query.fetch_objects(
            filters=wvc_query.Filter.by_property("codebase_name").equal(codebase_name),
            limit=1
        )
        if details.objects:
            manager.client.collections.get("CodebaseRegistry").data.update(
                uuid=details.objects[0].uuid,
                properties={"watcher_active": False}
            )

    return True, f"Watcher for '{codebase_name}' stopped."
