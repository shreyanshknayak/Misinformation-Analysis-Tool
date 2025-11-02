import os
import uuid
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 1. Import CORSMiddleware ---
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Import the core logic from your tool package
from image_search_tool.tool import generate_image_timeline

# --- Configuration ---

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

# Vertex AI Configuration
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")

# --- Client Initialization ---

# Initialize GCS client
try:
    storage_client = storage.Client()
except Exception as e:
    print(f"Warning: Could not initialize GCS client. Error: {e}")
    storage_client = None

# Initialize Vertex AI
try:
    if not GCP_PROJECT_ID:
        raise Exception("GCP_PROJECT_ID environment variable not set.")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    # Using Flash for speed and lower cost, ideal for summarization
    # Corrected model name
    summary_model = GenerativeModel(model_name="gemini-2.0-flash")
except Exception as e:
    print(f"Warning: Could not initialize Vertex AI. Error: {e}")
    summary_model = None

# --- FastAPI App ---

app = FastAPI(
    title="Image Timeline Tool API",
    description="An API to upload an image, perform a reverse image search, and generate a summary of its context.",
    version="2.0.0",
)

# --- 2. Add CORS Middleware ---

# Define the "origins" (domains) that are allowed to make requests
# Update this list with your actual frontend domains
origins = [
    "http://localhost",              # For local testing
    "http://localhost:3000",         # Common for React dev
    "http://localhost:8000",         # For local testing
    "https://your-tailwind-app-domain.com", # <-- REPLACE THIS
    # You can use "*" to allow all domains, but it's less secure
    # "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of origins that can make requests
    allow_credentials=True,         # Allow cookies (if you use them)
    allow_methods=["*"],            # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers
)

# --- Pydantic Response Models ---

class MatchedLink(BaseModel):
    """Defines a single entry in the timeline/matched links list."""
    id: int
    domain: str
    date: str
    url: str
    title: Optional[str] = None # 'tool.py' now provides this

class FinalResponse(BaseModel):
    """The final response model for the API."""
    summary: str
    matched_links: List[MatchedLink]

# --- Vertex AI Summarization ---

async def get_summary_from_vertex(context_data: dict) -> str:
    """
    Generates a summary of the image's context using Vertex AI.
    """
    if not summary_model:
        return "Summary generation is unavailable: Vertex AI client not initialized."

    try:
        # Create a clean context dict to send to the model
        # We exclude 'timeline_data' as it's redundant for the summary
        prompt_context = {
            "vision_analysis": context_data.get("vision_analysis"),
            "scraped_page_context": context_data.get("scraped_page_context")
        }
        
        # Serialize the context data to a JSON string
        context_json = json.dumps(prompt_context, indent=2, default=str)

        prompt = f"""
You are an AI assistant specializing in image provenance and context analysis. 
You will receive a JSON object with data from a Google reverse image search.
Your job is to generate a concise, 1-2 paragraph summary of what the image is, 
its main subject, and the context in which it appears online.
Mention the earliest known appearance if available from the 'scraped_page_context'.

Base your summary *only* on the provided JSON data. Do not invent information.

Here is the data:
{context_json}
"""

        # Generate the summary
        response = await summary_model.generate_content_async(prompt)
        
        if not response.candidates or not response.candidates[0].content.parts:
            raise Exception("No content in Vertex AI response.")
            
        return response.candidates[0].content.parts[0].text

    except Exception as e:
        print(f"Error during Vertex AI summarization: {e}")
        return f"Error: Could not generate summary. {str(e)}"

# --- API Endpoints ---

@app.post("/generate-timeline", response_model=FinalResponse)
async def generate_timeline_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image file, uploads it to GCS, performs a reverse
    image search, generates a context summary, and returns the
    summary and a list of matched links.
    """
    
    # Check server configuration
    if not GCS_BUCKET_NAME or not storage_client:
        raise HTTPException(
            status_code=500, 
            detail="Server is not configured for GCS uploads. Missing GCS_BUCKET_NAME or credentials."
        )
    if not summary_model:
        raise HTTPException(
            status_code=500, 
            detail="Server is not configured for summarization. Vertex AI not initialized."
        )

    try:
        # 1. Upload file to GCS (same as before)
        contents = await file.read()
        file_extension = os.path.splitext(file.filename)[1]
        blob_name = f"uploads/{uuid.uuid4()}{file_extension}"
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(contents, content_type=file.content_type)
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"

        # 2. Call your tool to get the rich context data
        # This is the full dictionary from 'tool.py'
        tool_output = generate_image_timeline(gcs_uri)

        # 3. Generate the summary
        summary = await get_summary_from_vertex(tool_output)

        # 4. Extract the formatted timeline
        matched_links = tool_output.get('timeline_data', [])

        # 5. Return the final, formatted response
        return FinalResponse(summary=summary, matched_links=matched_links)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"An internal error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Image Timeline API is running. POST to /generate-timeline"}