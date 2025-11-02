import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fact_check_claim(query: str) -> str:
    """
    Searches for fact-checked claims related to a given query using the Google Fact Check Tools API.
    
    Args:
        query: The textual query string to search for.
        
    Returns:
        A formatted string summarizing the fact-check results, including URLs and titles of articles.
        Returns an informative message if no claims are found or if an error occurs.
    """
    # 1. Correctly get the API key from the environment
    api_key = os.getenv("FACT_CHECK_API_KEY")
    if not api_key:
        return "Error: FACT_CHECK_API_KEY environment variable is not set."

    # 2. Correctly define the API endpoint URL
    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    # 3. Use the variables correctly in the request parameters
    params = {
        "query": query,
        "key": api_key  # Use the variable containing the API key
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        claims = data.get("claims", [])
        
        if not claims:
            return f"No fact-checked claims found for the query: \"{query}\"."
        
        results_list = []
        for claim in claims:
            claim_text = claim.get("text", "No claim text available.")
            
            claim_reviews = claim.get("claimReview", [])
            for review in claim_reviews:
                review_url = review.get("url", "No URL available.")
                review_title = review.get("title", "No title available.")
                review_rating = review.get("textualRating", "N/A")
                publisher_name = review.get("publisher", {}).get("name", "Unknown Publisher")
                
                # Format the information for each individual review
                results_list.append(
                    f"**Claim:** {claim_text}\n"
                    f"**Rating:** {review_rating}\n"
                    f"**Reviewed by:** {publisher_name}\n"
                    f"**Article Title:** {review_title}\n"
                    f"**Article URL:** {review_url}\n"
                )
        
        # Combine all the formatted results into a single, comprehensive string
        return "Fact-checking results:\n\n" + "\n---\n".join(results_list)
        
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An unexpected error occurred: {err}"