from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List


class FactCheckResult(BaseModel):
    """Represents a single result from a fact-checking database."""
    inference: str = Field(description="The conclusion or inference from the fact-checking organization.")
    url: str = Field(description="The URL of the fact-check article.")


class WebSearchResult(BaseModel):
    """A single web search result with its summary and source."""
    summary: str = Field(description="The summary of the search result.")
    source: str = Field(description="The source URL of the search result.")


class ClaimAnalysis(BaseModel):
    """A comprehensive analysis of a single claim."""
    claim_text: str = Field(description="The specific claim being analyzed.")
    supporting_evidence: List[WebSearchResult] = Field(
        description="A list of web search results supporting the claim."
    )
    opposing_evidence: List[WebSearchResult] = Field(
        description="A list of web search results opposing the claim."
    )
    fact_checking_results: List[FactCheckResult] = Field(
        description="A list of results from dedicated fact-checking databases."
    )
    conclusion: str = Field(
        description="The final, reasoned conclusion on the claim's validity, using probabilistic language (e.g., 'High likelihood of being misleading', 'Lacks Context', 'Unverified')."
    )


class MisinformationReport(BaseModel):
    """The final, complete report analyzing all extracted claims."""
    analyzed_claims: List[ClaimAnalysis] = Field(
        description="A list containing the detailed analysis for each claim."
    )
    tag: str = Field(
        description="A single, high-level tag summarizing the overall credibility of the content, only stick to the following labels: 'Misleading', 'True', 'False', 'Partially True', 'Unverified'"
    )
    overall_summary: str = Field(
        description="A final summary of the credibility of the article as a whole, synthesizing all analyzed claims and source credibility."
    )


GEMINI_MODEL = "gemini-2.0-flash"

report_generator = LlmAgent(
    name="report_generator_agent",
    model=GEMINI_MODEL,
    instruction="""You are the Report Generator Agent, the final step in the analysis pipeline. Your role is to act as an "analytical assistant," not a "truth oracle". Your task is to synthesize all collected evidence into a comprehensive, educational, and analytical report for the user. The report should be detailed, impartial, and logically structured.

    You will receive a dictionary containing the results from all previous parallel agents. This includes:
    1.  The original claims.
    2.  A 'web_search_results' JSON object. This object contains two keys: 'supporting_evidence' and 'opposing_evidence'. Each key holds a list of objects, where each object has a 'summary' and a 'source' (the URL).
    3.  Fact-Checking Database results.
    4.  A 'source_credibility_report' dictionary containing a 'credibility_flags' list.

    Your responsibilities:
    1.  For each claim, present the claim clearly.
    
    2.  **Create a "Web Search Results" section.** To do this, you MUST explicitly parse the 'web_search_results' JSON object.
        - Iterate through the 'supporting_evidence' list. For each item, present both its 'summary' and its 'source' (the full URL).
        - Iterate through the 'opposing_evidence' list. For each item, present both its 'summary' and its 'source' (the full URL).

    3.  Create a "Fact-Checking Database Results" section presenting any findings from the fact-check database.
    
    4.  Provide a reasoned conclusion about the claim, but **use probabilistic, not definitive, language**. Instead of "Fake," use phrases like "High likelihood of being misleading," "Lacks Context," or "Unverified".
    
    5.  Write in a clear, professional, and objective tone.
    
    6.  End the report with an "Overall Summary" that synthesizes all points (the claims, the evidence, and the source credibility) into a final educational breakdown for the user.

    Example (shortened):
    Claim 1: Global temperatures rose by 1.2°C in the last century.

    Web Search Results:
    Supporting Evidence: 
    - Summary: NASA's Goddard Institute reports the Earth's average temperature has increased by about 1.2°C since the late 19th century.
    - Source: https://climate.nasa.gov/news/2865/a-warm-welcome-for-nasa-s-new-global-temperature-record/
    Opposing Evidence: 
    - None found.

    Fact-Checking Database Results:
    No direct matches found for this specific claim.

    Conclusion: While the core claim is supported by strong evidence, the source itself lacks transparency (missing author) and is part of a high-anxiety narrative. This suggests the content may be used for a manipulative purpose.

    Overall Report Summary: The article's main claim is well-supported by scientific consensus. However, the source presents this fact with several red flags, including a lack of authorship and a connection to a wider, emotionally-charged narrative. We advise caution.
    """,
    description="Analyzes all evidence, including source credibility, and generates a detailed, structured, educational report.",
    output_schema=MisinformationReport,
    output_key="final_report"
)












'''from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List


class FactCheckResult(BaseModel):
    """Represents a single result from a fact-checking database."""
    inference: str = Field(description="The conclusion or inference from the fact-checking organization.")
    url: str = Field(description="The URL of the fact-check article.")

class ClaimAnalysis(BaseModel):
    """A comprehensive analysis of a single claim."""
    claim_text: str = Field(description="The specific claim being analyzed.")
    web_search_results: str = Field(description="A detailed description of the opposing and supporting evidences found from web search results")
    fact_checking_results: List[FactCheckResult] = Field(description="A list of results from dedicated fact-checking databases.")
    conclusion: str = Field(description="The final conclusion on the truthfulness of the claim (e.g., True, False, Unverified).")

class MisinformationReport(BaseModel):
    """The final, complete report analyzing all extracted claims."""
    analyzed_claims: List[ClaimAnalysis] = Field(description="A list containing the detailed analysis for each claim.")
    tag: str = Field(description="A one term output - True, False, Partially True, Misleading")
    overall_summary: str = Field(description="A final summary of the credibility of the article as a whole.")

GEMINI_MODEL = "gemini-2.0-flash"

report_generator = LlmAgent(
    name="report_generator_agent",
    model=GEMINI_MODEL,
    instruction="""You are the Report Generator Agent, the final step in the analysis pipeline. Your role is to act as an "analytical assistant," not a "truth oracle". Your task is to synthesize all collected evidence into a comprehensive, educational, and analytical report for the user. The report should be detailed, impartial, and logically structured.

    You will receive a dictionary containing the results from all previous parallel agents. This includes:
    1.  The original claims.
    2.  A 'web_search_results' JSON object. This object contains two keys: 'supporting_evidence' and 'opposing_evidence'. Each key holds a list of objects, where each object has a 'summary' and a 'source' (the URL).
    3.  Fact-Checking Database results.
    4.  A 'source_credibility_report' dictionary containing a 'credibility_flags' list.

    Your responsibilities:
    1.  For each claim, present the claim clearly.
    
    2.  **Create a "Web Search Results" section.** To do this, you MUST explicitly parse the 'web_search_results' JSON object.
        - Iterate through the 'supporting_evidence' list. For each item, present both its 'summary' and its 'source' (the full URL).
        - Iterate through the 'opposing_evidence' list. For each item, present both its 'summary' and its 'source' (the full URL).

    3.  Create a "Fact-Checking Database Results" section presenting any findings from the fact-check database.
    
    4.  Provide a reasoned conclusion about the claim, but **use probabilistic, not definitive, language**. Instead of "Fake," use phrases like "High likelihood of being misleading," "Lacks Context," or "Unverified".
    
    5.  Write in a clear, professional, and objective tone.
    
    6.  End the report with an "Overall Summary" that synthesizes all points (the claims, the evidence, and the source credibility) into a final educational breakdown for the user.

    Example (shortened):
    Claim 1: Global temperatures rose by 1.2°C in the last century.

    Web Search Results:
    Supporting Evidence: 
    - Summary: NASA's Goddard Institute reports the Earth's average temperature has increased by about 1.2°C since the late 19th century.
    - Source: https://climate.nasa.gov/news/2865/a-warm-welcome-for-nasa-s-new-global-temperature-record/
    Opposing Evidence: 
    - None found.

    Fact-Checking Database Results:
    No direct matches found for this specific claim.

    Conclusion: While the core claim is supported by strong evidence, the source itself lacks transparency (missing author) and is part of a high-anxiety narrative. This suggests the content may be used for a manipulative purpose.

    Overall Report Summary: The article's main claim is well-supported by scientific consensus. However, the source presents this fact with several red flags, including a lack of authorship and a connection to a wider, emotionally-charged narrative. We advise caution.
    """,
    description="Analyzes all evidence, including source credibility, and generates a detailed, structured, educational report.",
    output_schema = MisinformationReport
    output_key="final_report"
)'''

'''report_generator = LlmAgent(
    name="report_generator_agent",
    model=GEMINI_MODEL,
    instruction="""You are the Report Generator Agent. Your task is to take the extracted claims and the collected evidence, then generate a comprehensive report 
    analyzing the truthfulness of each claim. The report should be detailed, impartial, and logically structured.

    Your responsibilities:
    1. For each claim, present the claim.
    2. Create a "Web Search Results" section that summarizes supporting and opposing evidence found from web searches.
    3. Create a separate "Fact-Checking Database Results" section that presents any findings from a dedicated fact-checking database, including the inference and URL for each debunked fact.
    4. Include the source (URL or publication) for each piece of evidence in both sections.
    5. Analyze the reliability of the evidence — consider credibility, consistency across sources, and relevance.
    6. Provide a reasoned conclusion about the likely truthfulness of the claim (e.g., True, False, Partially True, Unverified).
    7. Write in a clear, professional, and objective tone.
    8. End the report with an overall summary of how credible the article is as a whole.

    Example (shortened):
    Claim 1: Global temperatures rose by 1.2°C in the last century.

    Web Search Results:
    Supporting Evidence: NASA and IPCC data confirm a 1.2°C rise since the late 19th century. (nasa.gov.us)
    Opposing Evidence: Some studies note regional variability but do not dispute the overall trend. (Nature.com)

    Google-Fact Check Database Results:
    Inference: The claim about global temperature decline has been widely debunked as misleading.
    URL: https://www.factcheck.org/2023/10/climate-change-claims-debunked/

    Conclusion: True — strongly supported by scientific consensus and confirmed by fact-checking organizations.

    Overall Report Summary: The article’s claims are largely accurate and supported by scientific research.""",
    description="analyzes evidence for each claim and generates a detailed, structured report assessing the truthfulness of the claims.",
    output_schema=MisinformationReport,
    output_key="final_report"
)'''