import json
import os
import hashlib
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Union, Dict
import asyncio
from datetime import datetime  # <-- NEW: For timestamping

# --- 1. .env path loading (no changes) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

this_file_path = Path(__file__)
root_dir = this_file_path.parent 
env_path = root_dir / ".env"
logger.info(f"Attempting to load .env file from: {env_path}")
load_dotenv(dotenv_path=env_path)

# --- 2. Imports (NEW additions) ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# --- GCP Client Imports (NEW additions) ---
import redis
from google.cloud import firestore
from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Your existing agents
try:
    from .sub_agents.claims_extractor_agent.agent import claims_extractor
    from .sub_agents.web_scraper_agent.agent import web_scraper
    from .sub_agents.report_generator_agent.agent import report_generator
    from .sub_agents.fact_checking_agent.agent import fact_checker  
    logger.info("✅ All sub-agents imported successfully.")
except ImportError as e:
    logger.error(f"❌ FAILED to import sub-agents. Check .env variables. Error: {e}")
    # This will cause the app to fail on startup, which is good.
    raise

# ==============================================================================
# --- 3. INITIALIZE GCP CLIENTS (MODIFIED) ---
# ==============================================================================

# --- L1 Cache: Memorystore (Redis) Client ---
try:
    redis_client = redis.Redis(
        host=os.environ["MEMORSTORE_HOST"],
        port=int(os.environ["MEMORSTORE_PORT"]),
        decode_responses=True  # Decode from bytes to strings
    )
    redis_client.ping()
    logger.info("✅ Connected to Memorystore (Redis)")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Memorystore: {e}", exc_info=True)
    redis_client = None

# --- Report Storage: Firestore Client ---
try:
    firestore_client = firestore.Client(database="misinfo-reports") # <-- Includes DB fix
    REPORT_COLLECTION = os.environ["FIRESTORE_REPORT_COLLECTION"]
    logger.info(f"✅ Connected to Firestore (collection: {REPORT_COLLECTION})")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Firestore: {e}", exc_info=True)
    firestore_client = None

# --- NEW: Vertex AI Client Initialization ---
try:
    # --- Make sure these are in your Cloud Run Env Variables ---
    PROJECT_ID = os.environ["PROJECT_ID"]
    REGION = os.environ["REGION"]
    vertexai.init(project=PROJECT_ID, location=REGION)

    # --- Embedding Model ---
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    logger.info("✅ Connected to Vertex AI (Embedding Model)")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Vertex AI: {e}", exc_info=True)
    embedding_model = None

# --- NEW: BigQuery Client Initialization ---
try:
    bq_client = bigquery.Client()
    BQ_TABLE_ID = "misinformation_logs.submissions" # Make sure this dataset/table exists
    logger.info(f"✅ Connected to BigQuery (table: {BQ_TABLE_ID})")
except Exception as e:
    logger.error(f"❌ FAILED to connect to BigQuery: {e}", exc_info=True)
    bq_client = None

# --- L1 Caching Constants ---
# Set TTL to 6 hours (60s * 60m * 6h)
L1_CACHE_TTL_SECONDS = 21600 


# ==============================================================================
# --- 4. Agent Definitions (Renumbered, no changes) ---
# ==============================================================================

parallel_research_agent = ParallelAgent(
    name="ParallelResearchAgent",
    sub_agents=[web_scraper, fact_checker],
    description="Runs web scraping and fact checking tasks in parallel to speed up evidence gathering."
)

root_agent = SequentialAgent(
    name="MisinformationAnalysisPipeline",
    sub_agents=[claims_extractor, parallel_research_agent, report_generator],
    description="A multi-agent pipeline that performs parallel research, and generates a report."
)

# ==============================================================================
# --- 5. Initialize the Runner (Renumbered, no changes) ---
# ==============================================================================

session_service = InMemorySessionService()
runner = Runner(
    app_name="misinformation-app",
    agent=root_agent,
    session_service=session_service,
)

# ==============================================================================
# --- 6. FastAPI App (Renumbered, no changes) ---
# ==============================================================================

app = FastAPI(
    title="Misinformation Analysis Agent",
    description="API endpoint for the multi-agent misinformation pipeline with L1 Caching and BQ Logging.",
    version="1.2.0" # Updated version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# --- 7. Async Helper Functions for Caching (MODIFIED) ---
# ==============================================================================

# These run sync (blocking) code in a thread to avoid blocking the server

async def get_report_from_firestore_async(report_id: str) -> Union[Dict[str, Any], None]:
    if not firestore_client:
        return None
    try:
        def _get_doc():
            doc_ref = firestore_client.collection(REPORT_COLLECTION).document(report_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        return await asyncio.to_thread(_get_doc)
    except Exception as e:
        logger.error(f"Error fetching from Firestore: {e}", exc_info=True)
        return None

async def save_report_to_firestore_async(report: Dict[str, Any]) -> str:
    def _set_doc():
        doc_ref = firestore_client.collection(REPORT_COLLECTION).document()
        doc_ref.set(report)
        return doc_ref.id
    return await asyncio.to_thread(_set_doc)

async def redis_get_async(key: str):
    return await asyncio.to_thread(redis_client.get, key)

async def redis_set_async(key: str, value: str):
    # This is where we set the 6-hour TTL
    return await asyncio.to_thread(redis_client.set, key, value, ex=L1_CACHE_TTL_SECONDS)

# --- NEW: Embedding and BigQuery Helper Functions ---

async def get_embedding_async(text: str):
    if not embedding_model:
        logger.error("Embedding model not initialized.")
        return None
    def _get_emb():
        # [0].values is the list of 768 floats
        return embedding_model.get_embeddings([text])[0].values
    return await asyncio.to_thread(_get_emb)

async def save_log_to_bigquery_async(report_object: Dict, hash: str, text: str):
    if not bq_client:
        logger.error("BigQuery client not initialized. Skipping log.")
        return
    
    try:
        # 1. Extract claims_text from the report object
        # Based on your logs, 'analyzed_claims' is a list of dicts
        claims_list = report_object.get('analyzed_claims', [])
        claims_text_list = [claim.get('claim_text', '') for claim in claims_list]
        full_claims_text = " ".join(claims_text_list)
        
        # 2. Generate embedding (only if we have text)
        embedding = None
        if full_claims_text:
            embedding = await get_embedding_async(full_claims_text)
        
        # 3. Build the row
        row_to_insert = {
            "timestamp": datetime.utcnow().isoformat(),
            "submission_hash": hash,
            "input_text": text,
            "report_summary": json.dumps(report_object),
            "claims_list_json": json.dumps(claims_list),
            "semantic_embedding": embedding if embedding else []
        }
        
        # 4. Save to BigQuery
        def _insert_row():
            errors = bq_client.insert_rows_json(BQ_TABLE_ID, [row_to_insert])
            if errors:
                logger.error(f"Failed to insert row to BigQuery: {errors}")
        
        await asyncio.to_thread(_insert_row)
        logger.info(f"💾 Saved log to BigQuery for hash: {hash}")

    except Exception as e:
        logger.error(f"Failed during BigQuery logging: {e}", exc_info=True)


# ==============================================================================
# --- 8. The Refactored Endpoint (MODIFIED) ---
# ==============================================================================

@app.get("/")
async def run_agent(text: str):
    """
    Accepts a text claim, checks L1 cache, logs to BQ, 
    runs the ADK Runner, and returns the result.
    """
    
    # --- 0. Health Check ---
    if not all([redis_client, firestore_client, bq_client, embedding_model]):
        logger.error("A backend client is not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Backend clients not initialized.")

    user_id = "local-user"
    processed_input = text
    input_hash = hashlib.sha256(processed_input.encode()).hexdigest()

    # --- 1. L1 Cache Check (Exact Match) ---
    try:
        cached_report_id = await redis_get_async(input_hash)
        if cached_report_id:
            logger.info(f"✅ L1 Cache HIT: {input_hash}")
            report = await get_report_from_firestore_async(cached_report_id)
            if report:
                # --- NEW: Log the cache hit to BigQuery ---
                await save_log_to_bigquery_async(report, input_hash, processed_input)
                return report  # Return immediately
            else:
                # This fixes a rare bug where Redis has a key but Firestore doesn't.
                logger.warning(f"⚠️ L1 Cache MISTAKE: Report ID {cached_report_id} not found in Firestore. Running full pipeline.")
    except Exception as e:
        logger.warning(f"⚠️ L1 Cache ERROR: {e}", exc_info=True)

    logger.info(f"❌ L1 Cache MISS: {input_hash}. Running full pipeline...")

    # --- 2. Full Pipeline (Cache Miss) ---
    # This is your original logic, now triggered on a miss.
    try:
        session = await session_service.create_session(
            app_name=runner.app_name, 
            user_id=user_id
        )
        logger.info(f"Created new session: {session.id}")
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        return {"error": f"Failed to create session: {str(e)}"}
    
    new_message = types.Content(role="user", parts=[types.Part(text=processed_input)])
    
    final_report_object = None
    report_string = None 
    
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id, 
            new_message=new_message
        ):
            
            if hasattr(event, 'author') and event.author == "report_generator_agent":
                logger.info(">>> Found event from 'report_generator_agent'!")
                try:
                    if event.content and event.content.parts:
                        report_string = event.content.parts[0].text
                    
                    if report_string:
                        logger.info(f"Found report string: {report_string[:50]}...")
                        final_report_object = json.loads(report_string)
                        break # We got our final report
                    else:
                        logger.warning("...but 'text' part was empty or missing.")
                except Exception as e:
                    logger.error(f"...but failed to parse. Error: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Unexpected error during agent run: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

    # --- 3. Cache Write (on Cache Miss) ---
    if final_report_object:
        try:
            # --- NEW: Log the new report to BigQuery BEFORE caching ---
            # This is the main log, creating the embedding for the first time
            await save_log_to_bigquery_async(final_report_object, input_hash, processed_input)

            # Save to Firestore first to get a permanent ID
            new_report_id = await save_report_to_firestore_async(final_report_object)
            logger.info(f"💾 Saved report to Firestore: {new_report_id}")
            
            # Now save that ID to Redis with our 6-hour TTL
            await redis_set_async(input_hash, new_report_id)
            logger.info(f"💾 Saved hash to L1 Cache (Redis) with {L1_CACHE_TTL_SECONDS}s TTL")
        
        except Exception as e:
            # Log the error, but don't fail the request.
            # The user should still get their report even if caching fails.
            logger.error(f"Failed to write to cache/BQ: {e}", exc_info=True)
        
        # Return the report we just generated
        return final_report_object
    else:
        logger.warning("Agent ran, but no final report was generated.")
        return {"error": "Agent ran, but no final report was generated."}





############################################################################################
import json
import os
import hashlib
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Union, Dict
import asyncio
from datetime import datetime
import mimetypes  # <-- No longer need base64

# --- 1. .env path loading (no changes) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

this_file_path = Path(__file__)
root_dir = this_file_path.parent
env_path = root_dir / ".env"
logger.info(f"Attempting to load .env file from: {env_path}")
load_dotenv(dotenv_path=env_path)

# --- 2. Imports (MODIFIED) ---
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# --- GCP Client Imports (MODIFIED) ---
import redis
from google.cloud import firestore
from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import vision  # <-- NEW: Import Google Cloud Vision
# Use an alias for vision types to avoid conflict with genai.types
from google.cloud.vision_v1 import types as vision_types

# Your existing agents
try:
    from .sub_agents.claims_extractor_agent.agent import claims_extractor
    from .sub_agents.web_scraper_agent.agent import web_scraper
    from .sub_agents.report_generator_agent.agent import report_generator
    from .sub_agents.fact_checking_agent.agent import fact_checker
    logger.info("✅ All sub-agents imported successfully.")
except ImportError as e:
    logger.error(f"❌ FAILED to import sub-agents. Check .env variables. Error: {e}")
    raise

# ==============================================================================
# --- 3. INITIALIZE GCP CLIENTS (MODIFIED) ---
# ==============================================================================

# --- L1 Cache: Memorystore (Redis) Client ---
try:
    redis_client = redis.Redis(
        host=os.environ["MEMORSTORE_HOST"],
        port=int(os.environ["MEMORSTORE_PORT"]),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("✅ Connected to Memorystore (Redis)")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Memorystore: {e}", exc_info=True)
    redis_client = None

# --- Report Storage: Firestore Client ---
try:
    firestore_client = firestore.Client(database="misinfo-reports")
    REPORT_COLLECTION = os.environ["FIRESTORE_REPORT_COLLECTION"]
    logger.info(f"✅ Connected to Firestore (collection: {REPORT_COLLECTION})")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Firestore: {e}", exc_info=True)
    firestore_client = None

# --- Vertex AI Client Initialization ---
try:
    PROJECT_ID = os.environ["PROJECT_ID"]
    REGION = os.environ["REGION"]
    vertexai.init(project=PROJECT_ID, location=REGION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    logger.info("✅ Connected to Vertex AI (Embedding Model)")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Vertex AI: {e}", exc_info=True)
    embedding_model = None

# --- BigQuery Client Initialization ---
try:
    bq_client = bigquery.Client()
    BQ_TABLE_ID = f"{PROJECT_ID}.misinformation_logs.submissions"
    logger.info(f"✅ Connected to BigQuery (table: {BQ_TABLE_ID})")
except Exception as e:
    logger.error(f"❌ FAILED to connect to BigQuery: {e}", exc_info=True)
    bq_client = None

# --- NEW: Google Cloud Vision (OCR) Client Initialization ---
try:
    vision_client = vision.ImageAnnotatorClient()
    logger.info("✅ Connected to Google Cloud Vision (OCR)")
except Exception as e:
    logger.error(f"❌ FAILED to connect to Cloud Vision: {e}", exc_info=True)
    vision_client = None


# --- L1 Caching Constants ---
L1_CACHE_TTL_SECONDS = 21600


# ==============================================================================
# --- 4. Agent Definitions (no changes) ---
# ==============================================================================

parallel_research_agent = ParallelAgent(
    name="ParallelResearchAgent",
    sub_agents=[web_scraper, fact_checker],
    description="Runs web scraping and fact checking tasks in parallel to speed up evidence gathering."
)

root_agent = SequentialAgent(
    name="MisinformationAnalysisPipeline",
    sub_agents=[claims_extractor, parallel_research_agent, report_generator],
    description="A multi-agent pipeline that performs parallel research, and generates a report."
)

# ==============================================================================
# --- 5. Initialize the Runner (no changes) ---
# ==============================================================================

session_service = InMemorySessionService()
runner = Runner(
    app_name="misinformation-app",
    agent=root_agent,
    session_service=session_service,
)

# ==============================================================================
# --- 6. FastAPI App (no changes) ---
# ==============================================================================

app = FastAPI(
    title="Misinformation Analysis Agent",
    description="API endpoint for the multi-agent misinformation pipeline with L1 Caching and BQ Logging.",
    version="1.4.0"  # Updated version for OCR pipeline
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# --- 7. Async Helper Functions (MODIFIED) ---
# ==============================================================================

async def get_report_from_firestore_async(report_id: str) -> Union[Dict[str, Any], None]:
    if not firestore_client:
        return None
    try:
        def _get_doc():
            doc_ref = firestore_client.collection(REPORT_COLLECTION).document(report_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        return await asyncio.to_thread(_get_doc)
    except Exception as e:
        logger.error(f"Error fetching from Firestore: {e}", exc_info=True)
        return None

async def save_report_to_firestore_async(report: Dict[str, Any]) -> str:
    def _set_doc():
        doc_ref = firestore_client.collection(REPORT_COLLECTION).document()
        doc_ref.set(report)
        return doc_ref.id
    return await asyncio.to_thread(_set_doc)

async def redis_get_async(key: str):
    return await asyncio.to_thread(redis_client.get, key)

async def redis_set_async(key: str, value: str):
    return await asyncio.to_thread(redis_client.set, key, value, ex=L1_CACHE_TTL_SECONDS)

async def get_embedding_async(text: str):
    if not embedding_model:
        logger.error("Embedding model not initialized.")
        return None
    def _get_emb():
        return embedding_model.get_embeddings([text])[0].values
    return await asyncio.to_thread(_get_emb)

# --- NEW: OCR Helper Function ---
async def run_ocr_on_image_async(file_bytes: bytes) -> str:
    if not vision_client:
        logger.error("Vision client not initialized. Skipping OCR.")
        return ""
    try:
        def _run_ocr():
            image = vision_types.Image(content=file_bytes)
            response = vision_client.text_detection(image=image)
            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")
            if response.full_text_annotation:
                return response.full_text_annotation.text
            return ""
        
        return await asyncio.to_thread(_run_ocr)
    except Exception as e:
        logger.error(f"Failed during OCR processing: {e}", exc_info=True)
        return ""

async def save_log_to_bigquery_async(report_object: Dict, hash: str, text: str):
    if not bq_client:
        logger.error("BigQuery client not initialized. Skipping log.")
        return

    try:
        claims_list = report_object.get('analyzed_claims', [])
        claims_text_list = [claim.get('claim_text', '') for claim in claims_list]
        full_claims_text = " ".join(claims_text_list)

        embedding = None
        if full_claims_text:
            embedding = await get_embedding_async(full_claims_text)

        row_to_insert = {
            "timestamp": datetime.utcnow().isoformat(),
            "submission_hash": hash,
            "input_text": text, # This is the original text prompt from the user
            "report_summary": json.dumps(report_object),
            "claims_list_json": json.dumps(claims_list),
            "semantic_embedding": embedding if embedding else []
        }

        def _insert_row():
            errors = bq_client.insert_rows_json(BQ_TABLE_ID, [row_to_insert])
            if errors:
                logger.error(f"Failed to insert row to BigQuery: {errors}")

        await asyncio.to_thread(_insert_row)
        logger.info(f"💾 Saved log to BigQuery for hash: {hash}")

    except Exception as e:
        logger.error(f"Failed during BigQuery logging: {e}", exc_info=True)


# ==============================================================================
# --- 8. The Refactored Endpoint (HEAVILY MODIFIED) ---
# ==============================================================================

@app.post("/")
async def run_agent(
    text: str = Form(""),
    file: UploadFile = File(None)
):
    """
    Accepts text and/or an image file. If an image is provided,
    it runs OCR to extract text and adds it to the prompt
    before running the analysis pipeline.
    """

    # --- 0. Health Check ---
    if not all([redis_client, firestore_client, bq_client, embedding_model, vision_client]): # <-- Added vision_client
        logger.error("A backend client is not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Backend clients not initialized.")

    user_id = "local-user"
    input_hash = None
    file_bytes = None
    
    # --- 1. Handle Input and Create Hash ---
    if file:
        logger.info(f"Received file: {file.filename}")
        file_bytes = await file.read()
        input_hash = hashlib.sha256(file_bytes).hexdigest()
    elif text:
        logger.info("Received text-only request.")
        input_hash = hashlib.sha256(text.encode()).hexdigest()
    else:
        raise HTTPException(status_code=400, detail="No text or file provided.")

    # --- 2. L1 Cache Check ---
    try:
        cached_report_id = await redis_get_async(input_hash)
        if cached_report_id:
            logger.info(f"✅ L1 Cache HIT: {input_hash}")
            report = await get_report_from_firestore_async(cached_report_id)
            if report:
                await save_log_to_bigquery_async(report, input_hash, text)
                return report
            else:
                logger.warning(f"⚠️ L1 Cache MISTAKE: Report ID {cached_report_id} not found. Running full pipeline.")
    except Exception as e:
        logger.warning(f"⚠️ L1 Cache ERROR: {e}", exc_info=True)

    logger.info(f"❌ L1 Cache MISS: {input_hash}. Running full pipeline...")

    # --- 3. NEW: OCR Pre-processing Step ---
    input_text_for_agent = text  # Start with the user's text
    
    if file_bytes:
        mime_type, _ = mimetypes.guess_type(file.filename or 'default.jpg')
        if mime_type and mime_type.startswith("image/"):
            logger.info("Image file detected, running OCR...")
            try:
                ocr_text = await run_ocr_on_image_async(file_bytes)
                if ocr_text:
                    logger.info(f"OCR extracted text: {ocr_text[:150]}...")
                    # Combine original text with OCR text for the agent
                    input_text_for_agent = f"{text}\n\n[Extracted text from image]:\n{ocr_text}"
                else:
                    logger.info("OCR found no text in image.")
            except Exception as e:
                logger.error(f"OCR processing failed: {e}", exc_info=True)
                # Fail gracefully, just use the original text
        else:
            logger.info(f"File received, but it is not an image ({mime_type}). Will not run OCR.")

    # --- 4. Full Pipeline (Cache Miss) ---
    try:
        session = await session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id
        )
        logger.info(f"Created new session: {session.id}")
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        return {"error": f"Failed to create session: {str(e)}"}

    # --- MODIFIED: Build Text-Only Message ---
    # The agent now *only* receives text (either original or text + OCR text)
    parts = [types.Part(text=input_text_for_agent)]
    new_message = types.Content(role="user", parts=parts)
    # --- End of modification ---

    final_report_object = None
    report_string = None

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=new_message
        ):
            if hasattr(event, 'author') and event.author == "report_generator_agent":
                logger.info(">>> Found event from 'report_generator_agent'!")
                try:
                    report_string = None
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                report_string = part.text
                                break
                    
                    if report_string:
                        logger.info(f"Found report string: {report_string[:50]}...")
                        final_report_object = json.loads(report_string)
                        break
                    else:
                        logger.warning("...but 'text' part was empty or missing.")
                except Exception as e:
                    logger.error(f"...but failed to parse. Error: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Unexpected error during agent run: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

    # --- 5. Cache Write (on Cache Miss) ---
    if final_report_object:
        try:
            # Log to BQ (pass the *original* user text, not the combined text)
            await save_log_to_bigquery_async(final_report_object, input_hash, text)

            new_report_id = await save_report_to_firestore_async(final_report_object)
            logger.info(f"💾 Saved report to Firestore: {new_report_id}")
            
            await redis_set_async(input_hash, new_report_id)
            logger.info(f"💾 Saved hash to L1 Cache (Redis) with {L1_CACHE_TTL_SECONDS}s TTL")
        
        except Exception as e:
            logger.error(f"Failed to write to cache/BQ: {e}", exc_info=True)
        
        return final_report_object
    else:
        logger.warning("Agent ran, but no final report was generated.")
        return {"error": "Agent ran, but no final report was generated."}

