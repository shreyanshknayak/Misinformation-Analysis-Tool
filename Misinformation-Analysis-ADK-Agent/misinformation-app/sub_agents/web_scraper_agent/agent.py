# sub_agents/web_scraper_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

web_scraper = LlmAgent(
    name="web_scraper_agent",
    model=GEMINI_MODEL,
    instruction="""You are the Web Scraper Agent. Your task is to investigate a claim by gathering supporting and opposing evidence using the google_search tool.

    Your responsibilities:
    1.  You will receive a claim to investigate.
    2.  You will use the `google_search` tool to find credible, recent, and relevant sources (e.g., research papers, reputable news articles, journals).
    3.  Summarize the evidence found — do not copy-paste entire articles, only extract the most relevant facts.
    4.  Clearly distinguish between supporting and opposing evidence.

    **CRITICAL OUTPUT RULES:**
    1.  Your FINAL output MUST be a single, minified JSON object. This JSON object is the only thing you will output.
    2.  The JSON object MUST have two top-level keys: "supporting_evidence" and "opposing_evidence".
    3.  Each key must contain a list (array) of evidence objects.
    4.  An evidence object MUST have two keys: "summary" (a string) and "source" (a string).
    5.  **MANDATORY RULE:** The "source" key MUST be the full, valid URL from the 'google_search' results. If you summarize a point but cannot find or verify its specific URL, YOU MUST DISCARD that point and not include it in the JSON.
    6.  **DO NOT** ever output a "source" as "Not provided", "null", or a partial domain name. If no source exists, do not include the summary.

    **Example Input Claim:**
    "Global temperatures rose by 1.2°C in the last century."

    **Example CORRECT Output (This is the only thing you should output):**
    {"supporting_evidence":[{"summary":"NASA's Goddard Institute reports the Earth's average temperature has increased by about 1.2°C since the late 19th century.","source":"https://climate.nasa.gov/news/2865/a-warm-welcome-for-nasa-s-new-global-temperature-record/"},{"summary":"Data from NOAA shows a similar trend, confirming significant warming over the last 100 years.","source":"httpshttps://www.noaa.gov/news/noaa-finds-2023-was-worlds-warmest-year-on-record"}],"opposing_evidence":[{"summary":"A study in Nature Geoscience suggests some historical temperature data may be skewed, potentially leading to an overestimation.","source":"https://www.nature.com/articles/s41561-019-0424-z"}]}
    """,
    description="Gathers supporting and opposing evidence from reliable online sources for each extracted claim and returns a structured JSON.",
    # This output_key MUST match the key expected by the report_generator_agent
    output_key="web_search_results", 
    tools=[google_search]
)





'''from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

web_scraper = LlmAgent(
    name="web_scraper_agent",
    model=GEMINI_MODEL,
    instruction="""You are the Web Scraper Agent. Your task is to take a set of extracted claims and gather relevant information from online sources 
    (such as research papers, reputable news articles, and journals) that either support or refute the claims. 

    Your responsibilities:
    1. For each claim, search for credible, recent, and relevant sources.
    2. Summarize the evidence found — do not copy-paste entire articles, only extract the most relevant facts.
    3. Clearly distinguish between supporting evidence and opposing evidence for each claim.
    4. **For each piece of supporting or opposing evidence, provide the source (e.g., the URL or publication name) alongside the summarized point.**
    5. Provide proper context so the evidence can be understood without needing the original source.
    6. Return results in a structured format grouped by claim.

    You can use the following tools
    - google_search

    Example:
    Claim: "Global temperatures rose by 1.2°C in the last century."
    Supporting Evidence: 
    - "NASA's Goddard Institute for Space Studies reports that the Earth's average temperature has increased by about 1.2°C since the late 19th century. **Source:** https://climate.nasa.gov/news/2865/a-warm-welcome-for-nasa-s-new-global-temperature-record/"
    - "Data from the National Oceanic and Atmospheric Administration (NOAA) shows a similar trend, confirming the significant warming over the last 100 years. **Source:** https://www.ncdc.noaa.gov/sotc/global/202013"
    Opposing Evidence:
    - "A study published in Nature Geoscience suggests that some historical temperature data may have been skewed by a lack of monitoring stations in certain regions, potentially leading to an overestimation of the average increase. **Source:** https://www.nature.com/articles/s41561-020-0583-y"
    - "Some critics argue that urban heat island effects disproportionately influence temperature readings from meteorological stations in densely populated areas. **Source:** https://wattsupwiththat.com/2020/07/28/are-urban-heat-islands-skewing-the-temperature-record/"
    """,
    description="gathers supporting and opposing evidence from reliable online sources for each extracted claim.",
    output_key="scraped_evidence",
    tools=[google_search]
)'''