import os
import logging
import google.generativeai as genai
import google.api_core.exceptions
from dotenv import load_dotenv
import asyncio
import re
import weaviate.classes.query as wvc_query

from weaviate_client import (
    create_weaviate_client,
    find_relevant_elements,
    get_element_details,
    update_element_properties,
    find_element_by_name,
    get_codebase_details,  # Import needed function
)

logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None
embedding_model_name = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GENERATION_MODEL_NAME = os.getenv(
            "GENERATION_MODEL_NAME", "models/gemini-2.0-flash-001"
        )
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
    except Exception as e:
        logger.error(f"Error initializing Gemini models in rag.py: {e}")
        model = None
        embedding_model_name = None
else:
    logger.warning("GEMINI_API_KEY not found in rag.py. LLM features disabled.")


# --- RAG Function ---
async def answer_codebase_question(
    query_text: str, client=None, tenant_id: str = None
) -> str:
    """
    Answers questions about the codebase using RAG:
    1. Finds relevant code elements using semantic search within a specific tenant and its dependencies.
    2. Constructs a prompt with the context and question.
    3. Calls a generative LLM to get the answer.
    """
    logger.info(f"Answering question: '{query_text}' for tenant '{tenant_id}'")

    # Ensure LLM models are available
    if not model or not embedding_model_name:
        return "ERROR: Generative or embedding model not configured in rag.py. Cannot answer question."

    # Ensure tenant_id is provided
    if not tenant_id:
        return "ERROR: tenant_id must be provided to answer codebase questions."

    client_created_internally = False
    weaviate_client = client

    try:
        # If no client was passed, create one internally
        if weaviate_client is None:
            logger.debug("answer_codebase_question: No client passed, creating one...")
            weaviate_client = create_weaviate_client()
            if not weaviate_client:
                logger.error(
                    "answer_codebase_question: Failed to create internal Weaviate client."
                )
                return "ERROR: Failed to create Weaviate client for RAG."
            client_created_internally = True

        # Ensure client is connected
        if not weaviate_client.is_connected():
            logger.debug(
                "answer_codebase_question: Client not connected, connecting..."
            )
            weaviate_client.connect()
            if not weaviate_client.is_connected():
                logger.error(
                    "answer_codebase_question: Failed to connect Weaviate client."
                )
                if client_created_internally:
                    weaviate_client.close()
                return "ERROR: Failed to connect Weaviate client for RAG."
        logger.debug("answer_codebase_question: Weaviate client connected.")

        # Ensure primary tenant exists
        elements_collection = weaviate_client.collections.get("CodeElement")
        tenant_exists = await asyncio.to_thread(
            elements_collection.tenants.exists, tenant_id
        )
        if not tenant_exists:
            logger.error(f"Tenant '{tenant_id}' does not exist.")
            if client_created_internally:
                weaviate_client.close()
            return f"ERROR: Codebase (tenant) '{tenant_id}' does not exist."

        # --- Dependency Handling ---
        tenant_ids_to_query = [tenant_id]
        try:
            # Fetch dependencies for the primary tenant (wrap sync call)
            codebase_details = await asyncio.to_thread(
                get_codebase_details, weaviate_client, tenant_id
            )
            if codebase_details and codebase_details.get("dependencies"):
                dependencies = codebase_details["dependencies"]
                logger.info(f"Found dependencies for RAG: {dependencies}")
                # Check if dependency tenants exist before adding (wrap sync call)
                for dep_name in dependencies:
                    dep_exists = await asyncio.to_thread(
                        elements_collection.tenants.exists, dep_name
                    )
                    if dep_exists:
                        tenant_ids_to_query.append(dep_name)
                    else:
                        logger.warning(
                            f"Dependency codebase (tenant) '{dep_name}' not found in Weaviate. Skipping for RAG."
                        )
        except Exception as dep_e:
            logger.error(f"Error fetching dependencies for RAG query: {dep_e}")
        # --- End Dependency Handling ---

        # 1. Find relevant context using semantic search across tenants
        logger.debug(
            f"answer_codebase_question: Finding relevant elements via semantic search across tenants {tenant_ids_to_query}..."
        )
        # find_relevant_elements is synchronous, call directly
        primary_context_elements = find_relevant_elements(
            weaviate_client,
            tenant_ids_to_query,
            query_text,
            limit=5,  # Increased limit slightly for multi-tenant
        )
        if not primary_context_elements:
            logger.warning(
                f"answer_codebase_question: No primary relevant context found across tenants {tenant_ids_to_query}."
            )
            return "Could not find relevant context in the codebase or its dependencies to answer the question."

        # 2. Enhance context by fetching related elements (callers, callees, parent) within the element's specific tenant
        logger.debug(
            "answer_codebase_question: Enhancing context with related elements..."
        )
        enhanced_context = []
        processed_uuids = set()

        for elem in primary_context_elements:
            elem_uuid = elem.get("uuid")
            elem_tenant = elem.get("_tenant_id", tenant_id)  # Get tenant from result
            if not elem_uuid or elem_uuid in processed_uuids:
                continue

            elem_props = elem.get("properties", {})
            # Add tenant info to context if multiple tenants were queried
            if len(tenant_ids_to_query) > 1:
                elem_props["_codebase_source"] = elem_tenant
            enhanced_context.append(elem_props)
            processed_uuids.add(elem_uuid)

            try:
                # Fetch object with references within the element's specific tenant (wrap sync call)
                related_refs = await asyncio.to_thread(
                    elements_collection.with_tenant(
                        elem_tenant
                    ).query.fetch_object_by_id,  # Use element's tenant
                    uuid=elem_uuid,
                    return_references=[
                        wvc_query.QueryReference(
                            link_on="calls_function",
                            return_properties=[
                                "name",
                                "element_type",
                                "llm_description",
                                "code_snippet",
                            ],
                        ),
                        wvc_query.QueryReference(
                            link_on="method_of",
                            return_properties=[
                                "name",
                                "element_type",
                                "llm_description",
                                "code_snippet",
                            ],
                        ),
                    ],
                )
                if related_refs and related_refs.references:
                    for ref_type, refs in related_refs.references.items():
                        for ref_obj in refs.objects:
                            if ref_obj.uuid not in processed_uuids:
                                ref_props = ref_obj.properties
                                ref_props["relation_type"] = f"Related ({ref_type})"
                                # Add source codebase if needed
                                if len(tenant_ids_to_query) > 1:
                                    ref_props["_codebase_source"] = (
                                        elem_tenant  # Assume related refs are in same tenant
                                    )
                                enhanced_context.append(ref_props)
                                processed_uuids.add(ref_obj.uuid)
            except Exception as ref_e:
                logger.warning(
                    f"Failed to fetch related references for {elem_uuid} in tenant '{elem_tenant}': {ref_e}"
                )

        # 3. Format enhanced context for prompt
        context_str = "\n\n---\n\n".join(
            [
                f"Source Codebase: {elem.get('_codebase_source', tenant_id)}\n"  # Show source codebase
                f"Relation: {elem.get('relation_type', 'Primary Match')}\n"
                f"File: {elem.get('file_path', 'N/A')}\n"
                f"Type: {elem.get('element_type', 'N/A')}\n"
                f"Name: {elem.get('name', 'N/A')}\n"
                f"Description: {elem.get('llm_description', '').strip()}\n"
                f"Code Snippet:\n```python\n{elem.get('code_snippet', '').strip()}\n```"
                for elem in enhanced_context[:7]  # Limit context size
            ]
        )

        # 4. Construct prompt
        prompt = (
            "You are an AI assistant analyzing Python code.\n"
            f"You are analyzing the codebase named '{tenant_id}' and potentially its dependencies: {', '.join(tenant_ids_to_query)}.\n"  # Clarify context scope
            "Based *only* on the following code snippets and descriptions extracted from the relevant codebase(s), answer the user's question.\n"
            "If context comes from multiple codebases, mention the source codebase for clarity.\n"
            "Do not use any prior knowledge. If the provided context is insufficient to answer the question, state that clearly.\n\n"
            "CONTEXT:\n"
            "-------\n"
            f"{context_str}\n"
            "-------\n\n"
            f"QUESTION: {query_text}\n\n"
            "ANSWER:"
        )

        logger.debug(f"--- RAG Prompt ---\n{prompt}\n------------------")

        # 5. Call generative LLM
        logger.debug("answer_codebase_question: Generating answer with LLM...")
        try:
            # Wrap synchronous LLM call
            response = await asyncio.to_thread(model.generate_content, prompt)
            answer = response.text.strip()
            logger.info("answer_codebase_question: LLM generation successful.")
            return answer
        except google.api_core.exceptions.GoogleAPIError as api_e:
            logger.exception(
                f"answer_codebase_question: Google API exception during LLM generation: {api_e}"
            )
            return f"ERROR: Google API error during generation: {api_e}"
        except Exception as llm_e:
            logger.exception(
                f"answer_codebase_question: Exception during LLM generation: {llm_e}"
            )
            return f"ERROR: Failed to generate answer using LLM: {llm_e}"

    except Exception as e:
        logger.exception(f"answer_codebase_question: Unexpected error: {e}")
        return f"ERROR: An unexpected error occurred: {e}"
    finally:
        if (
            client_created_internally
            and weaviate_client
            and weaviate_client.is_connected()
        ):
            logger.debug(
                "answer_codebase_question: Closing internally created Weaviate client."
            )
            weaviate_client.close()


async def refine_element_description(client, tenant_id: str, element_uuid: str) -> bool:
    """
    Refines the LLM description of a code element using broader context
    (callers, callees, siblings, related variables) within a specific tenant.
    Returns True if successful, False otherwise.
    """
    logger.info(
        f"Refining description for element UUID: {element_uuid} in tenant '{tenant_id}'"
    )

    if not model:
        logger.error(
            "refine_element_description: LLM model not configured. Cannot refine."
        )
        return False

    if not tenant_id:
        logger.error("refine_element_description: tenant_id must be provided.")
        return False

    weaviate_client = client

    try:
        elements_collection = weaviate_client.collections.get("CodeElement")
        if not weaviate_client or not weaviate_client.is_connected():
            logger.error(
                "refine_element_description: Weaviate client not provided or not connected."
            )
            return False

        tenant_exists = await asyncio.to_thread(
            elements_collection.tenants.exists, tenant_id
        )
        if not tenant_exists:
            logger.error(f"Tenant '{tenant_id}' does not exist for refinement.")
            return False

        target_element = await asyncio.to_thread(
            get_element_details, weaviate_client, tenant_id, element_uuid
        )
        if not target_element:
            logger.error(
                f"refine_element_description: Target element {element_uuid} not found in tenant '{tenant_id}'."
            )
            return False

        target_props = target_element.properties
        target_name = target_props.get("name", "Unknown")
        target_type = target_props.get("element_type", "Unknown")
        target_code = target_props.get("code_snippet", "")
        target_file = target_props.get("file_path", "")
        current_desc = target_props.get("llm_description", "")
        target_readable_id = target_props.get("readable_id", "N/A")
        target_signature = target_props.get("signature", "N/A")

        if target_type not in ["function", "class"]:
            logger.debug(
                f"Skipping refinement for non-function/class element: {target_name} ({target_type})"
            )
            return True

        context_parts = {}

        try:
            target_with_refs = await asyncio.to_thread(
                elements_collection.with_tenant(tenant_id).query.fetch_object_by_id,
                uuid=element_uuid,
                return_references=[
                    wvc_query.QueryReference(
                        link_on="calls_function",
                        return_properties=["name", "element_type", "llm_description"],
                    )
                ],
            )
            if target_with_refs and target_with_refs.references:
                callees = target_with_refs.references.get("calls_function", [])
                if callees and callees.objects:
                    context_parts["Callees"] = "\n".join(
                        [
                            f"- {c.properties.get('name', '?')} ({c.properties.get('element_type', '?')}): {c.properties.get('llm_description', 'No description')}"
                            for c in callees.objects
                        ]
                    )
        except Exception as ref_e:
            logger.warning(
                f"Failed to fetch callees for {element_uuid} in tenant '{tenant_id}': {ref_e}"
            )

        try:
            response_callers = await asyncio.to_thread(
                elements_collection.with_tenant(tenant_id).query.fetch_objects,
                filters=wvc_query.Filter.by_ref_count("calls_function").greater_than(0)
                & wvc_query.Filter.by_reference("calls_function").contains_any(
                    [element_uuid]
                ),
                limit=5,
                return_properties=["name", "element_type", "llm_description"],
            )
            if response_callers.objects:
                context_parts["Callers"] = "\n".join(
                    [
                        f"- {c.properties.get('name', '?')} ({c.properties.get('element_type', '?')}): {c.properties.get('llm_description', 'No description')}"
                        for c in response_callers.objects
                    ]
                )
        except Exception as ref_e:
            logger.warning(
                f"Failed to fetch callers for {element_uuid} in tenant '{tenant_id}': {ref_e}"
            )

        if target_file:
            try:
                response_siblings = await asyncio.to_thread(
                    find_element_by_name,
                    weaviate_client,
                    [tenant_id],
                    file_path=target_file,
                    element_type=target_type,
                    limit=10,
                )
                siblings = [s for s in response_siblings if str(s.uuid) != element_uuid]
                if siblings:
                    context_parts[f"Siblings ({target_type}s) in same file"] = (
                        "\n".join(
                            [
                                f"- {s.properties.get('name', '?')}: {s.properties.get('llm_description', 'No description')}"
                                for s in siblings
                            ]
                        )
                    )
            except Exception as sib_e:
                logger.warning(
                    f"Failed to fetch siblings for {element_uuid} in {target_file} (tenant '{tenant_id}'): {sib_e}"
                )

        try:
            variables = set(
                re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()", target_code)
            )
            common_keywords = {
                "self",
                "return",
                "def",
                "class",
                "if",
                "else",
                "elif",
                "for",
                "while",
                "try",
                "except",
                "finally",
                "with",
                "import",
                "from",
                "as",
                "pass",
                "None",
                "True",
                "False",
            }
            variables = variables - common_keywords
            related_vars_context = []
            if variables:
                logger.debug(f"Found potential variables: {variables}")
                for var in list(variables)[:3]:
                    related_snippets = find_relevant_elements(
                        weaviate_client,
                        [tenant_id],
                        query_text=f"Usage of variable {var}",
                        limit=2,
                        distance=0.6,
                    )
                    for rel in related_snippets:
                        if (
                            str(rel.get("uuid")) != element_uuid
                            and len(rel["properties"].get("code_snippet", "")) < 200
                        ):
                            related_vars_context.append(
                                f"- Related to '{var}' in {rel['properties'].get('file_path', '?')}:\n  ```python\n{rel['properties'].get('code_snippet', '')}\n  ```"
                            )
            if related_vars_context:
                context_parts["Related Variable Usage"] = "\n".join(
                    related_vars_context
                )

        except Exception as var_e:
            logger.warning(
                f"Failed during related variable search for {element_uuid} in tenant '{tenant_id}': {var_e}"
            )

        context_summary = ""
        for title, content in context_parts.items():
            if content:
                context_summary += f"\n\n{title}:\n{content}"

        prompt = (
            f"Refine the description for the following Python {target_type} named '{target_name}' within the codebase '{tenant_id}'.\n"
            f"Current Description: {current_desc}\n\n"
            f"Code Snippet:\n```python\n{target_code}\n```\n"
        )
        if context_summary:
            prompt += f"\nConsider the following context:{context_summary}\n"
        prompt += (
            f"\nReadable ID: {target_readable_id}\n"
            f"Signature: {target_signature}\n"
            "\nProvide a concise, improved, one-sentence description focusing on its primary purpose and interactions:"
        )

        logger.debug(
            f"--- Refinement Prompt for {target_name} ---\n{prompt}\n------------------"
        )

        max_retries = 3
        retry_delay = 2.0
        refined_desc = None
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Attempt {attempt + 1} to generate refined description for {target_name}"
                )
                refinement_response = await asyncio.to_thread(
                    model.generate_content, prompt
                )
                refined_desc = refinement_response.text.strip()
                logger.info(
                    f"LLM refinement successful for {target_name} on attempt {attempt + 1}."
                )
                break
            except google.api_core.exceptions.ResourceExhausted as rate_limit_e:
                logger.warning(
                    f"Rate limit hit during refinement for {target_name} (Attempt {attempt + 1}/{max_retries}): {rate_limit_e}. Retrying in {retry_delay}s..."
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Max retries reached for {target_name}. Refinement failed due to rate limiting."
                    )
                    refined_desc = None
                    break
            except google.api_core.exceptions.GoogleAPIError as api_e:
                logger.warning(
                    f"Google API error during description refinement for {target_name}: {api_e}"
                )
                refined_desc = None
                break
            except Exception as llm_e:
                logger.warning(
                    f"LLM refinement generation failed for {target_name}: {llm_e}"
                )
                refined_desc = None
                break

        if refined_desc is None:
            logger.error(
                f"Refinement failed for {target_name} after all attempts or due to non-retryable error."
            )
            return False

        if not refined_desc or refined_desc == current_desc:
            logger.info(f"No refinement needed or generated for {target_name}.")
            return True
        logger.info(f"Refined description for {target_name}: {refined_desc}")

        properties_to_update = target_props.copy()
        properties_to_update["llm_description"] = refined_desc
        update_success = await asyncio.to_thread(
            update_element_properties,
            weaviate_client,
            tenant_id,
            element_uuid,
            properties_to_update,
        )
        if not update_success:
            logger.error(
                f"Failed to update refined description for {element_uuid} in tenant '{tenant_id}'"
            )
            return False

        return True

    except Exception as e:
        logger.exception(
            f"Unexpected error during description refinement for {element_uuid} in tenant '{tenant_id}': {e}"
        )
        return False


# --- LLM Summary Generation ---
async def generate_codebase_summary(client, codebase_name: str) -> str:
    """
    Generates a concise summary for a codebase using key elements from its tenant.
    """
    logger.info(f"Generating summary for codebase (tenant): '{codebase_name}'")

    if not model:
        logger.error(
            "generate_codebase_summary: LLM model not configured. Cannot summarize."
        )
        return "Error: LLM model not configured."

    weaviate_client = client
    if not weaviate_client or not weaviate_client.is_connected():
        logger.error(
            "generate_codebase_summary: Weaviate client not provided or not connected."
        )
        return "Error: Weaviate client unavailable."

    try:
        elements_collection = weaviate_client.collections.get("CodeElement")

        # Ensure tenant exists (wrap sync call)
        tenant_exists = await asyncio.to_thread(
            elements_collection.tenants.exists, codebase_name
        )
        if not tenant_exists:
            logger.error(
                f"Tenant '{codebase_name}' does not exist for summary generation."
            )

            return f"Error: Codebase '{codebase_name}' not found."

        # Fetch top-level functions and classes (adjust filters as needed) (wrap sync call)
        logger.debug(f"Fetching key elements for codebase summary: {codebase_name}")
        response = await asyncio.to_thread(
            elements_collection.with_tenant(codebase_name).query.fetch_objects,
            filters=(
                wvc_query.Filter.by_property("element_type").contains_any(
                    ["function", "class"]
                )
            ),
            limit=20,
            return_properties=[
                "name",
                "element_type",
                "llm_description",
                "docstring",
                "file_path",
            ],
        )

        if not response.objects:
            logger.warning(
                f"No key elements found for codebase summary: {codebase_name}"
            )
            return "Codebase scanned, but no key functions or classes found to generate a summary."

        # Format context for summary prompt
        context_items = []
        for obj in response.objects:
            props = obj.properties
            desc = (
                props.get("llm_description")
                or props.get("docstring")
                or "No description available."
            )
            context_items.append(
                f"- {props.get('element_type', '?')} '{props.get('name', '?')}' in '{props.get('file_path', '?')}': {desc.strip()}"
            )

        context_str = "\n".join(context_items)

        # Construct summary prompt
        prompt = (
            f"Generate a concise, one-paragraph summary for the Python codebase named '{codebase_name}'.\n"
            "Focus on the codebase's main purpose and key components based *only* on the following list of functions and classes found within it:\n\n"
            "KEY ELEMENTS:\n"
            "-------------\n"
            f"{context_str}\n"
            "-------------\n\n"
            "SUMMARY:"
        )

        logger.debug(
            f"--- Codebase Summary Prompt for {codebase_name} ---\n{prompt}\n------------------"
        )

        # Call LLM for Summary (wrap sync call)
        try:
            summary_response = await asyncio.to_thread(model.generate_content, prompt)
            summary = summary_response.text.strip()
            logger.info(f"Successfully generated summary for codebase {codebase_name}.")
            return summary
        except google.api_core.exceptions.GoogleAPIError as api_e:
            logger.exception(
                f"Google API exception during summary generation for {codebase_name}: {api_e}"
            )
            return f"Error generating summary (API Error): {api_e}"
        except Exception as llm_e:
            logger.exception(
                f"Exception during summary generation for {codebase_name}: {llm_e}"
            )
            return f"Error generating summary: {llm_e}"

    except Exception as e:
        logger.exception(
            f"Unexpected error during summary generation for {codebase_name}: {e}"
        )
        return "Error: An unexpected error occurred during summary generation."
