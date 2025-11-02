import json
import os
import hashlib
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Union, Dict
import asyncio
from datetime import datetime
import mimetypes

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
# --- NEW: Import Gemini classes ---
from vertexai.generative_models import GenerativeModel, Part 

# --- REMOVED: Cloud Vision imports are no longer needed ---
# from google.cloud import vision
# from google.cloud.vision_v1 import types as vision_types

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

    # --- NEW: Vertex AI Multimodal (Gemini) Client Initialization ---
    multimodal_model = GenerativeModel("gemini-2.0-flash")
    logger.info("✅ Connected to Vertex AI (Gemini Multimodal Model)")

except Exception as e:
    logger.error(f"❌ FAILED to connect to Vertex AI: {e}", exc_info=True)
    embedding_model = None
    multimodal_model = None

# --- BigQuery Client Initialization ---
try:
    bq_client = bigquery.Client()
    BQ_TABLE_ID = f"{PROJECT_ID}.misinformation_logs.submissions"
    logger.info(f"✅ Connected to BigQuery (table: {BQ_TABLE_ID})")
except Exception as e:
    logger.error(f"❌ FAILED to connect to BigQuery: {e}", exc_info=True)
    bq_client = None

# --- REMOVED: Cloud Vision Client ---

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
    version="1.5.0"  # Updated version for Multimodal pipeline
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

# --- NEW: Multimodal Analysis Helper Function (replaces OCR) ---
async def run_multimodal_analysis_async(file_bytes: bytes, mime_type: str) -> str:
    if not multimodal_model:
        logger.error("Multimodal model not initialized. Skipping analysis.")
        return ""
    
    try:
        def _run_analysis():
            # This prompt instructs Gemini to do all three tasks
            prompt_text = (
                "You are a media analysis expert. Analyze the provided media (image, video, or audio) "
                "and provide a detailed, single-text-blob response. Your response must include:\n"
                "1.  **Transcription:** Transcribe any spoken audio in the media. If no audio, state 'No audio detected'.\n"
                "2.  **OCR:** Extract and transcribe any text visible in the media. If no text, state 'No text detected'.\n"
                "3.  **Description:** A detailed description of the key visual events or the main subject of the image/video.\n\n"
                "Combine all of this information into a single, comprehensive text response."
            )
            
            # Create the media part using the Vertex AI library
            media_part = Part.from_data(data=file_bytes, mime_type=mime_type)
            
            # Send the text prompt and the media part to Gemini
            response = multimodal_model.generate_content([prompt_text, media_part])
            
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return ""

        return await asyncio.to_thread(_run_analysis)
    except Exception as e:
        logger.error(f"Failed during multimodal analysis: {e}", exc_info=True)
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
    Accepts text and/or media files. If a file is provided,
    it runs multimodal analysis (OCR, Transcription, Description)
    and adds the extracted text to the prompt before running the pipeline.
    """

    # --- 0. Health Check ---
    if not all([redis_client, firestore_client, bq_client, embedding_model, multimodal_model]): # <-- Added multimodal_model
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

    # --- 3. NEW: Multimodal Pre-processing Step ---
    input_text_for_agent = text  # Start with the user's text
    
    if file_bytes:
        mime_type, _ = mimetypes.guess_type(file.filename or 'default.media')
        if mime_type is None:
            mime_type = 'application/octet-stream' # Best guess
        
        # Check if it's a media type Gemini can handle
        if mime_type.startswith("image/") or mime_type.startswith("video/") or mime_type.startswith("audio/"):
            logger.info(f"{mime_type} file detected, running multimodal analysis...")
            try:
                # This one function now handles OCR, description, and transcription
                analysis_text = await run_multimodal_analysis_async(file_bytes, mime_type)
                if analysis_text:
                    logger.info(f"Multimodal analysis extracted text: {analysis_text[:150]}...")
                    # Combine original text with the new analysis text
                    input_text_for_agent = f"{text}\n\n[Full media analysis (description, OCR, transcription)]:\n{analysis_text}"
                else:
                    logger.info("Multimodal analysis returned no text.")
            except Exception as e:
                logger.error(f"Multimodal analysis processing failed: {e}", exc_info=True)
                # Fail gracefully, just use the original text
        else:
            logger.info(f"File received, but it is not a supported media type ({mime_type}). Will not run analysis.")

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
    # The agent now *only* receives text (either original or text + analysis text)
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

