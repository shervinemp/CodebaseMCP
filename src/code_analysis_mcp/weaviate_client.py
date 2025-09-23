import weaviate
import datetime
import weaviate.classes.query as wvc_query
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    ReferenceProperty,
)
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
)

import logging
from typing import List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


from . import config

def create_weaviate_client():
    """Creates a Weaviate client, connecting to a URL from env vars if available."""
    logger.debug("Attempting to create Weaviate client...")
    load_dotenv()

    if config.WEAVIATE_URL:
        logger.info(f"Connecting to Weaviate at URL: {config.WEAVIATE_URL}")
        client = weaviate.connect_to_custom(
            http_host=config.WEAVIATE_URL.split(":")[-2].strip("/"),
            http_port=config.WEAVIATE_URL.split(":")[-1],
            http_secure=False,
            grpc_host=config.WEAVIATE_URL.split(":")[-2].strip("/"),
            grpc_port=50051,
            grpc_secure=False,
        )
    else:
        logger.info("WEAVIATE_URL not set, connecting to local Weaviate instance.")
        client = weaviate.connect_to_local()

    logger.info("Weaviate client object created.")
    return client


def create_schema(client):
    """Creates the CodeFile, CodeElement, and CodebaseRegistry collections."""
    collections = client.collections
    logger.info("Attempting to create/verify Weaviate schema...")

    # --- CodeFile Schema ---
    codefile_class_name = "CodeFile"
    logger.info(f"Schema: Checking existence of class '{codefile_class_name}'...")
    exists = collections.exists(codefile_class_name)
    logger.info(f"Schema: Class '{codefile_class_name}' exists: {exists}")
    if not exists:
        logger.info(f"Schema: Attempting to create class: {codefile_class_name}")
        try:
            collections.create(
                name=codefile_class_name,
                description="Represents a code file",
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="path",
                        data_type=DataType.TEXT,
                        description="The path to the code file",
                    ),
                    Property(
                        name="last_modified",
                        data_type=DataType.DATE,
                        description="The last modified time of the code file",
                    ),
                ],
            )
            logger.info(f"Schema: Successfully created class: {codefile_class_name}")
        except Exception as create_e:
            logger.exception(
                f"Schema: FAILED to create class {codefile_class_name}: {create_e}"
            )
            raise
    else:
        logger.info(
            f"Schema: Class '{codefile_class_name}' already exists, skipping creation."
        )

    # --- CodeElement Schema ---
    codeelement_class_name = "CodeElement"
    logger.info(f"Schema: Checking existence of class '{codeelement_class_name}'...")
    exists = collections.exists(codeelement_class_name)
    logger.info(f"Schema: Class '{codeelement_class_name}' exists: {exists}")
    if not exists:
        logger.info(f"Schema: Attempting to create class: {codeelement_class_name}")
        try:
            collections.create(
                name=codeelement_class_name,
                description="Represents a code element (function, class, variable, import, call, etc.)",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="element_type", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="start_line", data_type=DataType.INT),
                    Property(name="end_line", data_type=DataType.INT),
                    Property(name="code_snippet", data_type=DataType.TEXT),
                    Property(name="docstring", data_type=DataType.TEXT),
                    Property(name="parameters", data_type=DataType.TEXT_ARRAY),
                    Property(name="return_type", data_type=DataType.TEXT),
                    Property(name="signature", data_type=DataType.TEXT),
                    Property(name="readable_id", data_type=DataType.TEXT),
                    Property(name="parent_scope_uuid", data_type=DataType.TEXT),
                    Property(name="llm_description", data_type=DataType.TEXT),
                    Property(name="user_clarification", data_type=DataType.TEXT),
                    Property(name="decorators", data_type=DataType.TEXT_ARRAY),
                    Property(name="attribute_accesses", data_type=DataType.TEXT_ARRAY),
                    Property(name="base_class_names", data_type=DataType.TEXT_ARRAY),
                    Property(
                        name="last_modified",
                        data_type=DataType.DATE,
                    ),
                ],
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                references=[
                    ReferenceProperty(
                        name="defined_in_file", target_collection=codefile_class_name
                    ),
                    ReferenceProperty(
                        name="method_of", target_collection=codeelement_class_name
                    ),
                    ReferenceProperty(
                        name="defines_method", target_collection=codeelement_class_name
                    ),
                    ReferenceProperty(
                        name="calls_function", target_collection=codeelement_class_name
                    ),
                    ReferenceProperty(
                        name="called_by", target_collection=codeelement_class_name
                    ),
                    ReferenceProperty(
                        name="defines_variable",
                        target_collection=codeelement_class_name,
                    ),
                ],
            )
            logger.info(f"Schema: Successfully created class: {codeelement_class_name}")
        except Exception as create_e:
            logger.exception(
                f"Schema: FAILED to create class {codeelement_class_name}: {create_e}"
            )
            raise
    else:
        logger.info(
            f"Schema: Class '{codeelement_class_name}' already exists, skipping creation."
        )

    # --- CodebaseRegistry Schema ---
    codebaseregistry_class_name = "CodebaseRegistry"
    logger.info(
        f"Schema: Checking existence of class '{codebaseregistry_class_name}'..."
    )
    exists = collections.exists(codebaseregistry_class_name)
    logger.info(f"Schema: Class '{codebaseregistry_class_name}' exists: {exists}")
    if not exists:
        logger.info(
            f"Schema: Attempting to create class: {codebaseregistry_class_name}"
        )
        try:
            collections.create(
                name=codebaseregistry_class_name,
                description="Tracks scanned codebases and their status",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="codebase_name",
                        data_type=DataType.TEXT,
                        description="Unique name for the codebase",
                    ),
                    Property(
                        name="directory",
                        data_type=DataType.TEXT,
                        description="Absolute path of the codebase directory",
                    ),
                    Property(
                        name="scan_timestamp",
                        data_type=DataType.DATE,
                        description="Timestamp of the last successful scan/update",
                    ),
                    Property(
                        name="summary",
                        data_type=DataType.TEXT,
                        description="LLM-generated summary of the codebase",
                    ),
                    Property(
                        name="status",
                        data_type=DataType.TEXT,
                        description='Scanning status (e.g., "Scanning", "Summarizing", "Ready", "Error")',
                    ),
                    Property(
                        name="watcher_active",
                        data_type=DataType.BOOL,
                        description="Indicates if a file watcher is active for this codebase",
                    ),
                    Property(
                        name="dependencies",
                        data_type=DataType.TEXT_ARRAY,
                        description="List of codebase names this codebase depends on.",
                    ),
                ],
            )
            logger.info(
                f"Schema: Successfully created class: {codebaseregistry_class_name}"
            )
        except Exception as create_e:
            logger.exception(
                f"Schema: FAILED to create class {codebaseregistry_class_name}: {create_e}"
            )
            raise
    else:
        logger.info(
            f"Schema: Class '{codebaseregistry_class_name}' already exists, skipping creation."
        )

    logger.info("Schema: Schema creation/verification function completed.")


# --- Batching Functions ---


def add_objects_batch(client, objects: list[dict], class_name: str, tenant_id: str):
    """Adds multiple objects to a class using the client's batch, specific to a tenant."""
    logger.info(
        f"Adding {len(objects)} objects manually to batch for class '{class_name}' in tenant '{tenant_id}'..."
    )
    results = None
    objects_to_create = [
        weaviate.classes.data.DataObject(
            properties=obj_data.get("properties", {}),
            vector=obj_data.get("vector"),
            uuid=obj_data.get("uuid"),
        )
        for obj_data in objects
    ]

    logger.debug(
        f"Calling client.batch.create_objects for {len(objects_to_create)} objects..."
    )
    collection = client.collections.get(class_name)
    results = collection.with_tenant(tenant_id).data.insert_many(objects_to_create)
    logger.debug("client.batch.create_objects call completed.")

    if results.has_errors:
        logger.error(
            f"Errors occurred during batch insertion for '{class_name}' in tenant '{tenant_id}': {results.errors}"
        )
        if isinstance(results.errors, dict):
            for index, error_message in results.errors.items():
                logger.error(f"  Object index {index}: {error_message}")
        return {
            "status": "error",
            "message": "Batch insertion failed for some objects.",
            "errors": results.errors,
            "has_errors": True,
        }
    else:
        successful_count = len(objects_to_create)
        logger.info(
            f"Successfully inserted {successful_count} objects into '{class_name}' in tenant '{tenant_id}'."
        )
        return {"status": "success", "count": successful_count, "has_errors": False}


def add_references_batch(client, references: list[dict], tenant_id: str):
    """Adds multiple references using the client's batch, specific to a tenant."""
    logger.info(
        f"Adding {len(references)} references to batch in tenant '{tenant_id}'..."
    )
    with client.batch.dynamic() as batch:
        for ref_data in references:
            batch.add_reference(
                from_uuid=ref_data["from_uuid"],
                from_collection=ref_data["from_collection"],
                from_property=ref_data["from_property"],
                to=ref_data["to"],
                tenant=tenant_id,
            )
    logger.info(
        f"Batch context exited after adding {len(references)} references in tenant '{tenant_id}'."
    )
    return {"status": "success", "count": len(references)}


# --- Query Functions (CodeFile/CodeElement - Tenant Specific) ---
def get_all_code_files(client, tenant_id: str):
    """Retrieves all CodeFile objects for a specific tenant and returns a dict mapping path to last_modified."""
    logger.info(f"Fetching all code files from Weaviate for tenant '{tenant_id}'.")
    try:
        collection = client.collections.get("CodeFile")
        response = collection.with_tenant(tenant_id).query.fetch_objects(
            limit=10000, return_properties=["path", "last_modified"]
        )
        files_data = {}
        for obj in response.objects:
            try:
                stored_datetime = obj.properties["last_modified"]
                if stored_datetime.tzinfo is None:
                    stored_datetime = stored_datetime.replace(
                        tzinfo=datetime.timezone.utc
                    )
                files_data[obj.properties["path"]] = stored_datetime
            except Exception as parse_e:
                logger.error(
                    f"Error processing last_modified for {obj.properties['path']} in tenant '{tenant_id}': {parse_e}. Value: {obj.properties.get('last_modified')}"
                )
        logger.info(f"Fetched {len(files_data)} code files for tenant '{tenant_id}'.")
        logger.debug(
            f"get_all_code_files returning data for tenant '{tenant_id}': {files_data}"
        )
        return files_data
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} fetching code files for tenant '{tenant_id}': {e}"
        )
        return {}


def find_element_by_name(
    client,
    tenant_ids: List[str],
    element_name: str | None = None,
    file_path: str | None = None,
    element_type: str | None = None,
    parent_scope_uuid: str | None = None,
    element_class="CodeElement",
    limit=10,
):
    """Finds elements by name, file path, element type, and/or parent scope across multiple tenants."""
    all_results = []
    logger.info(
        f"Finding element across tenants {tenant_ids} by name='{element_name}', file_path='{file_path}', element_type='{element_type}', parent_scope='{parent_scope_uuid}', class='{element_class}', limit={limit}"
    )

    filters: list[wvc_query.Filter] = []
    if element_name is not None:
        filters.append(wvc_query.Filter.by_property("name").equal(element_name))
    if file_path is not None:
        filters.append(wvc_query.Filter.by_property("file_path").equal(file_path))
    if element_type is not None:
        filters.append(wvc_query.Filter.by_property("element_type").equal(element_type))
    if parent_scope_uuid is not None:
        filters.append(
            wvc_query.Filter.by_property("parent_scope_uuid").equal(parent_scope_uuid)
        )

    if not filters:
        logger.warning(
            f"find_element_by_name called without any filters for tenants {tenant_ids}."
        )
        return []

    combined_filter = (
        filters[0] if len(filters) == 1 else wvc_query.Filter.all_of(filters)
    )

    try:
        collection = client.collections.get(element_class)
        for tenant_id in tenant_ids:
            logger.debug(f"Querying tenant: {tenant_id}")
            try:
                response = collection.with_tenant(tenant_id).query.fetch_objects(
                    limit=limit,
                    filters=combined_filter,
                    return_properties=[
                        "name",
                        "element_type",
                        "file_path",
                        "start_line",
                        "end_line",
                        "code_snippet",
                        "docstring",
                        "parameters",
                        "return_type",
                        "signature",
                        "readable_id",
                        "decorators",
                        "attribute_accesses",
                        "parent_scope_uuid",
                        "llm_description",
                        "user_clarification",
                        "base_class_names",
                    ],
                )
                logger.info(
                    f"Found {len(response.objects)} elements matching criteria for name='{element_name}' in tenant '{tenant_id}'."
                )
                for obj in response.objects:
                    obj.properties["_tenant_id"] = tenant_id
                all_results.extend(response.objects)
            except Exception as tenant_e:
                logger.error(f"Error querying tenant '{tenant_id}': {tenant_e}")

        logger.info(
            f"Total elements found across tenants {tenant_ids}: {len(all_results)}. Applying limit {limit}."
        )
        return all_results[:limit]

    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} finding element '{element_name}' in class '{element_class}' across tenants {tenant_ids}: {e}"
        )
        return []


def get_element_details(
    client, tenant_id: str, element_uuid, element_class="CodeElement"
):
    """Retrieves all properties and optionally the vector for a specific element by its UUID within a specific tenant."""
    logger.info(
        f"Getting details for element UUID '{element_uuid}' in class '{element_class}' for tenant '{tenant_id}'"
    )
    try:
        collection = client.collections.get(element_class)
        obj = collection.with_tenant(tenant_id).query.fetch_object_by_id(
            uuid=element_uuid, include_vector=True
        )
        logger.info(
            f"Successfully retrieved details for UUID '{element_uuid}' in tenant '{tenant_id}'."
        )
        return obj
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} getting details for element UUID '{element_uuid}' in class '{element_class}' for tenant '{tenant_id}': {e}"
        )
        return None


# --- Update Function (CodeElement - Tenant Specific) ---
def update_element_properties(
    client,
    tenant_id: str,
    uuid: str,
    properties_to_update: dict,
    element_class="CodeElement",
):
    """Updates specific properties for a given element UUID within a specific tenant, automatically adding last_modified."""
    properties_to_update["last_modified"] = datetime.datetime.now(datetime.timezone.utc)
    logger.info(
        f"Updating properties for element UUID '{uuid}' in class '{element_class}' for tenant '{tenant_id}' with: {properties_to_update}"
    )
    try:
        collection = client.collections.get(element_class)
        collection.with_tenant(tenant_id).data.update(
            uuid=uuid,
            properties=properties_to_update,
        )
        logger.info(
            f"Successfully updated properties for UUID '{uuid}' in tenant '{tenant_id}'."
        )
        return True
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} updating properties for element UUID '{uuid}' in tenant '{tenant_id}': {e}"
        )
        return False


# --- CodebaseRegistry Functions (Global - No Tenant) ---


def add_codebase_registry_entry(
    client,
    codebase_name: str,
    directory: str,
    status: str,
    summary: str = "",
    dependencies: List[str] = None,
) -> str | None:
    """Adds a new codebase entry to the CodebaseRegistry. Returns UUID on success, None on failure."""
    logger.info(f"Adding codebase '{codebase_name}' to CodebaseRegistry.")
    collection = None
    try:
        logger.debug("Getting CodebaseRegistry collection...")
        collection = client.collections.get("CodebaseRegistry")
        logger.debug(f"Preparing entry data for {codebase_name}...")
        entry_data = {
            "codebase_name": codebase_name,
            "directory": directory,
            "status": status,
            "summary": summary,
            "scan_timestamp": datetime.datetime.now(datetime.timezone.utc),
            "watcher_active": False,
            "dependencies": dependencies or [],
        }
        logger.debug(f"Attempting insert for {codebase_name}...")
        result = collection.data.insert(properties=entry_data)
        logger.debug(
            f"Insert call completed for {codebase_name}. Result type: {type(result)}"
        )
        # The result object *is* the UUID, just convert it directly.
        result_uuid_str = str(result)
        logger.info(
            f"Successfully added codebase '{codebase_name}' with UUID {result_uuid_str}."
        )
        logger.debug(f"add_codebase_registry_entry returning UUID: {result_uuid_str}")
        return result_uuid_str
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} adding codebase '{codebase_name}' to registry: {e}"
        )
        return None


def update_codebase_registry(client, codebase_name: str, updates: dict):
    """Updates an existing codebase entry in the CodebaseRegistry by codebase_name."""
    logger.info(
        f"Updating codebase '{codebase_name}' in CodebaseRegistry with: {updates}"
    )
    try:
        collection = client.collections.get("CodebaseRegistry")
        response = collection.query.fetch_objects(
            filters=wvc_query.Filter.by_property("codebase_name").equal(codebase_name),
            limit=1,
        )
        if not response.objects:
            logger.error(
                f"Codebase '{codebase_name}' not found in registry for update."
            )
            return False

        codebase_uuid = response.objects[0].uuid
        if "scan_timestamp" not in updates:
            updates["scan_timestamp"] = datetime.datetime.now(datetime.timezone.utc)

        collection.data.update(uuid=codebase_uuid, properties=updates)
        logger.info(
            f"Successfully updated codebase '{codebase_name}' (UUID: {codebase_uuid})."
        )
        return True
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} updating codebase '{codebase_name}' in registry: {e}"
        )
        return False


def get_codebase_details(client, codebase_name: str) -> dict | None:
    """Retrieves details for a specific codebase from the CodebaseRegistry."""
    logger.info(
        f"Getting details for codebase '{codebase_name}' from CodebaseRegistry."
    )
    try:
        collection = client.collections.get("CodebaseRegistry")
        response = collection.query.fetch_objects(
            filters=wvc_query.Filter.by_property("codebase_name").equal(codebase_name),
            limit=1,
            return_properties=[
                "codebase_name",
                "directory",
                "summary",
                "status",
                "scan_timestamp",
                "watcher_active",
                "dependencies",
            ],
        )
        if response.objects:
            details = response.objects[0].properties
            if isinstance(details.get("scan_timestamp"), datetime.datetime):
                details["scan_timestamp"] = details["scan_timestamp"].isoformat()
            logger.info(f"Found details for codebase '{codebase_name}'.")
            return details
        else:
            logger.warning(f"Codebase '{codebase_name}' not found in registry.")
            return None
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} getting details for codebase '{codebase_name}': {e}"
        )
        return None


def get_all_codebases(client) -> list[dict]:
    """Retrieves all codebases from the CodebaseRegistry."""
    logger.info("Getting all codebases from CodebaseRegistry.")
    try:
        collection = client.collections.get("CodebaseRegistry")
        response = collection.query.fetch_objects(
            limit=1000,
            return_properties=[
                "codebase_name",
                "directory",
                "summary",
                "status",
                "scan_timestamp",
                "watcher_active",
                "dependencies",
            ],
        )
        codebases = []
        for obj in response.objects:
            props = obj.properties
            if isinstance(props.get("scan_timestamp"), datetime.datetime):
                props["scan_timestamp"] = props["scan_timestamp"].isoformat()
            codebases.append(props)
        logger.info(f"Retrieved {len(codebases)} codebases from registry.")
        return codebases
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} getting all codebases from registry: {e}"
        )
        return []


def delete_codebase_registry_entry(client, codebase_name: str) -> bool:
    """Deletes a codebase entry from the CodebaseRegistry by codebase_name."""
    logger.info(f"Deleting codebase '{codebase_name}' from CodebaseRegistry.")
    try:
        collection = client.collections.get("CodebaseRegistry")
        response = collection.data.delete_many(
            where=wvc_query.Filter.by_property("codebase_name").equal(codebase_name)
        )
        deleted_count = response.successful
        if deleted_count > 0:
            logger.info(
                f"Successfully deleted {deleted_count} entry for codebase '{codebase_name}'."
            )
            return True
        elif response.matches == 0:
            logger.warning(
                f"Codebase '{codebase_name}' not found in registry for deletion."
            )
            return True
        else:
            logger.error(
                f"Failed to delete codebase '{codebase_name}'. Failed count: {response.failed}"
            )
            return False
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} deleting codebase '{codebase_name}' from registry: {e}"
        )
        return False


# --- Deletion Functions (Tenant Specific) ---
def delete_elements_by_file_path(
    client, tenant_id: str, file_path, element_class="CodeElement"
):
    """Deletes all elements of a given class associated with a specific file path within a specific tenant."""
    logger.info(
        f"Deleting elements for file '{file_path}' in class '{element_class}' for tenant '{tenant_id}'"
    )
    try:
        collection = client.collections.get(element_class)
        response = collection.with_tenant(tenant_id).data.delete_many(
            where=wvc_query.Filter.by_property("file_path").equal(file_path)
        )
        logger.info(
            f"Deletion result for file '{file_path}' in tenant '{tenant_id}': Matches={response.matches}, Successful={response.successful}, Failed={response.failed}"
        )
        return response.successful > 0 or response.matches == 0
    except Exception as e:
        logger.exception(
            f"Exception type {type(e).__name__} deleting elements for file '{file_path}' in class '{element_class}' for tenant '{tenant_id}': {e}"
        )
        return False


def delete_code_file(client, tenant_id: str, file_path):
    """Deletes a CodeFile object based on its path within a specific tenant."""
    logger.info(f"Deleting CodeFile for path '{file_path}' in tenant '{tenant_id}'")
    try:
        collection = client.collections.get("CodeFile")
        response = collection.with_tenant(tenant_id).data.delete_many(
            where=wvc_query.Filter.by_property("path").equal(file_path)
        )
        logger.info(
            f"Deletion result for CodeFile '{file_path}' in tenant '{tenant_id}': Matches={response.matches}, Successful={response.successful}, Failed={response.failed}"
        )
        return response.successful > 0 or response.matches == 0
    except Exception as e:
        logger.error(
            f"Error deleting CodeFile for path '{file_path}' in tenant '{tenant_id}': {e}"
        )
        return False


def delete_tenant(client, tenant_id: str) -> bool:
    """Deletes a specific tenant from CodeFile and CodeElement collections."""
    logger.warning(
        f"Attempting to delete tenant '{tenant_id}' from CodeFile and CodeElement collections."
    )
    success = True
    try:
        # Delete from CodeElement
        element_collection = client.collections.get("CodeElement")
        if element_collection.tenants.exists(tenant_id):
            logger.info(f"Deleting tenant '{tenant_id}' from CodeElement...")
            element_collection.tenants.remove([tenant_id])
            logger.info(f"Tenant '{tenant_id}' removed from CodeElement.")
        else:
            logger.info(
                f"Tenant '{tenant_id}' does not exist in CodeElement. Skipping removal."
            )

        # Delete from CodeFile
        file_collection = client.collections.get("CodeFile")
        if file_collection.tenants.exists(tenant_id):
            logger.info(f"Deleting tenant '{tenant_id}' from CodeFile...")
            file_collection.tenants.remove([tenant_id])
            logger.info(f"Tenant '{tenant_id}' removed from CodeFile.")
        else:
            logger.info(
                f"Tenant '{tenant_id}' does not exist in CodeFile. Skipping removal."
            )

    except UnexpectedStatusCodeError as e:
        if e.status_code == 404:
            logger.warning(
                f"Tenant '{tenant_id}' not found during deletion attempt (status 404). Assuming already deleted."
            )
        else:
            logger.exception(f"Error deleting tenant '{tenant_id}': {e}")
            success = False
    except Exception as e:
        logger.exception(f"Unexpected error deleting tenant '{tenant_id}': {e}")
        success = False
    return success


# --- Semantic Search Function (Tenant Specific) ---
def find_relevant_elements(  # Keep synchronous for now, RAG handles threading
    client,
    tenant_ids: List[str],
    query_text: str,
    element_class="CodeElement",
    limit=5,
    alpha=0.5,
):
    """Finds relevant elements using hybrid search based on query text across multiple tenants."""
    logger.info(
        f"Starting hybrid search across tenants {tenant_ids} for query: '{query_text}', class='{element_class}', limit={limit}, alpha={alpha}"
    )
    all_results = []
    try:
        import google.generativeai as genai

        if not config.EMBEDDING_MODEL_NAME:
            logger.error("Hybrid search: Embedding model name not configured.")
            return []

        query_vector = genai.embed_content(
            model=config.EMBEDDING_MODEL_NAME,
            content=query_text,
            task_type="RETRIEVAL_QUERY",
        ).get("embedding")

        if not query_vector:
            logger.error(f"Failed to generate query vector for '{query_text}'.")
            return []

        collection = client.collections.get(element_class)

        for tenant_id in tenant_ids:
            logger.debug(f"Querying tenant with hybrid search: {tenant_id}")
            response = collection.with_tenant(tenant_id).query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=limit,
                alpha=alpha,
                return_metadata=wvc_query.MetadataQuery(score=True),
                return_properties=[
                    "name", "element_type", "file_path", "start_line", "end_line",
                    "code_snippet", "docstring", "parameters", "return_type",
                    "signature", "readable_id", "decorators", "attribute_accesses",
                    "parent_scope_uuid", "llm_description", "user_clarification",
                    "base_class_names",
                ],
            )

            if response and response.objects:
                for obj in response.objects:
                    all_results.append({
                        "uuid": str(obj.uuid),
                        "properties": obj.properties,
                        "score": obj.metadata.score if obj.metadata else None,
                        "_tenant_id": tenant_id,
                    })

        # Sort aggregated results by score (descending) and apply overall limit
        all_results.sort(key=lambda x: x.get("score") or 0, reverse=True)
        final_results = all_results[:limit]

        logger.info(f"Hybrid search completed. Found {len(final_results)} final results.")
        return final_results

    except Exception as e:
        logger.exception(f"Unexpected error during hybrid search for '{query_text}': {e}")
        return []
