import os
from google.adk.agents import LlmAgent
from google.adk.tools.langchain_tool import LangchainTool
from langchain_scraperapi.tools import ScraperAPITool
from dotenv import load_dotenv

load_dotenv()

# Set the environment variable for the API key
#os.environ["SCRAPERAPI_API_KEY"] = "081d57156aae92f6dab45cc12dbf9663"

# Instantiate the LangChain tool
scraper_tool_instance = ScraperAPITool()

# This is the key change: Wrap the LangChain tool with LangchainTool
# You must explicitly provide the name and description.
scraper_tool = LangchainTool(tool=scraper_tool_instance)

GEMINI_MODEL = "gemini-2.0-flash"

claims_extractor = LlmAgent(
    name="claims_extractor_agent",
    model = GEMINI_MODEL,
    instruction="""You are the Claims Extractor Agent. Your task is to carefully read the input article or text and identify clear, factual, and verifiable claims. 
    A claim is a statement that expresses something that can be proven true or false (e.g., statistics, events, scientific findings, policy decisions, or cause-effect relationships).

    Your responsibilities:
    1. Extract only meaningful claims — avoid opinions, rhetorical questions, or vague statements. 
    2. Break down complex sentences into individual, standalone claims if needed. 
    3. For each claim, provide a concise, self-contained sentence without additional commentary. 
    4. Return the claims as a numbered list, where each item is one claim.

    If the given material is a url, scrape the contents of the url first using the following tool and then extract the claims:
    - scraper_tool

    Example:
    Input text: "Experts say global temperatures rose by 1.2°C in the last century, and this increase is linked to human activities."
    Output:
    1. Global temperatures rose by 1.2°C in the last century.  
    2. The increase in global temperatures is linked to human activities.""",
    description="identifies and extracts clear, verifiable claims from the input article or text.",
    output_key = "extracted_claims",
    tools=[scraper_tool]
)