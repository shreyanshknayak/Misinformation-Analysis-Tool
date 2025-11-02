import os
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from .fact_check_tool import fact_check_claim # Import your new tool
from dotenv import load_dotenv

load_dotenv()

# Create an instance of your custom tool
fact_check_tool_instance = FunctionTool(fact_check_claim)

# Define your ADK agent
fact_checker = Agent(
    model="gemini-2.0-flash", 
    name="FactCheckerAgent",
    description="Analyzes the results from the Google Fact Check Tools API to provide a brief, structured summary for each fact-checked claim.",
    instruction="""Use the 'fact_check_claim' tool to verify the accuracy of user claims.
    Based on the tool's output, provide a brief, structured summary for each result in the following format:
    Inference: (infer from the title, review, and rating)
    URL: (the URL from the fact-checker results)
    """,
    tools=[fact_check_tool_instance] 
)