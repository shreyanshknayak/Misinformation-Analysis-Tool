import requests
import json
import re
import time
from datetime import timezone
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import tldextract
from google.cloud import vision

# --- Part 1: Reverse Image Search from GCS (Updated) ---
def web_detection_from_gcs(gcs_uri: str):
    """
    Performs web detection on a GCS image and returns rich context.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = gcs_uri
    response = client.web_detection(image=image)
    web_detection = response.web_detection

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors"
        )
    
    result = {
        "full_matching_images": [p.url for p in web_detection.full_matching_images],
        "pages_with_matching_images": [p.url for p in web_detection.pages_with_matching_images],
        "visually_similar_images": [img.url for img in web_detection.visually_similar_images],
        # Added: Google's direct analysis of the image
        "best_guess_labels": [label.label for label in web_detection.best_guess_labels],
        "web_entities": [{"entity_id": e.entity_id, "description": e.description, "score": e.score} for e in web_detection.web_entities]
    }
    return result

# --- Part 2: Context Building & Data Extraction (Updated) ---
def fetch_url(url, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageTimelineBot/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def extract_page_context(html: str):
    """
    Extracts date, title, and meta description from HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    date = None
    title = soup.title.string if soup.title else None
    
    description_tag = soup.find('meta', attrs={'name': 'description'})
    description = description_tag.get('content') if description_tag else None

    # --- Date extraction logic (unchanged) ---
    for ld in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(ld.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                for key in ("datePublished", "dateCreated", "uploadDate"):
                    if key in item and item[key]:
                        date = dateparser.parse(item[key])
                        break
            if date: break
        except Exception: continue
    
    if not date:
        meta_props = ['article:published_time', 'og:published_time']
        for prop in meta_props:
            tag = soup.find('meta', property=prop)
            if tag and tag.get('content'):
                date = dateparser.parse(tag.get('content'))
                break
    
    if not date:
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            date = dateparser.parse(time_tag.get('datetime'))
    # --- End date logic ---

    return {
        "date": date,
        "title": title.strip() if title else None,
        "description": description.strip() if description else None
    }

def normalize_datetime(dt):
    if dt is None: return None
    if dt.tzinfo is not None: return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def build_scraped_context_from_vision(vision_result: dict):
    """
    Scrapes context (date, title, desc) from URLs found in vision results.
    """
    # Use pages_with_matching_images for broad context
    urls = set(vision_result.get('pages_with_matching_images', []))
    
    # Also add full_matching_images (often the original source)
    urls.update(vision_result.get('full_matching_images', []))
    
    page_context_list = []
    # Limit to 20 pages for speed
    for url in list(urls)[:20]:
        html = fetch_url(url)
        if not html: continue
        
        try:
            context = extract_page_context(html)
            host = tldextract.extract(url).registered_domain
            page_context_list.append({
                "url": url, 
                "host": host, 
                "date": context.get("date"),
                "title": context.get("title"),
                "description": context.get("description")
            })
            time.sleep(0.2) # Be polite
        except Exception:
            continue
            
    # Sort by date (earliest first)
    return sorted(page_context_list, key=lambda x: (x['date'] is None, normalize_datetime(x['date'])))

# --- Part 3: Main Tool Function & Final Formatting (Updated) ---
def generate_image_timeline(image_gcs_uri: str):
    """
    Main tool function to be called by the API.
    Returns a rich dictionary with timeline data, scraped context,
    and direct vision analysis.
    """
    if not image_gcs_uri.startswith("gs://"):
        raise ValueError("Invalid input: image_gcs_uri must be a valid GCS URI.")

    # 1. Get all data from Google Vision API
    vision_results = web_detection_from_gcs(image_gcs_uri)
    
    # 2. Scrape the web pages for context
    scraped_context_data = build_scraped_context_from_vision(vision_results)
    
    # 3. Format the simple timeline (as your API expects)
    timeline_data = []
    for i, item in enumerate(scraped_context_data):
        if item["date"]:
            timeline_data.append({
                "id": i + 1,
                "domain": item["host"],
                "date": item["date"].strftime("%Y-%m-%d"),
                "url": item["url"],
                "title": item["title"] # Add title to timeline
            })

    # 4. Prepare the final, rich response object
    result = {
        'timeline_data': timeline_data,
        'scraped_page_context': scraped_context_data,
        'vision_analysis': {
             "best_guess_labels": vision_results.get("best_guess_labels"),
             "web_entities": vision_results.get("web_entities"),
             "visually_similar_images": vision_results.get("visually_similar_images")
        }
    }
    
    return result