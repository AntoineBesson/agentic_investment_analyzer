# In main.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# Load environment variables from .env file
load_dotenv()

# --- TOOL DEFINITIONS ---
# You can keep your tool imports as they were
from crewai_tools import TavilySearchResults
from tools.financial_tools import YFinanceTool
from tools.scraping_tools import ScrapeWebsiteTool

search_tool = TavilySearchResults()
yfinance_tool = YFinanceTool()
scrape_tool = ScrapeWebsiteTool()

# --- AGENT DEFINITIONS ---

# Agent 1: Company Researcher (Specialist)
researcher = Agent(
  role='Company Research Analyst',
  goal='Gather comprehensive and up-to-date information about a specified company, its competitors, and its market position.',
  backstory="""An expert in scouring the internet for corporate information. 
  You are adept at using search tools to find official websites, press releases, and any relevant news.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool, scrape_tool]
)

# Agent 2: Financial Analyst (Specialist)
financial_analyst = Agent(
  role='Quantitative Financial Analyst',
  goal='Analyze financial data and key performance indicators for a given company using its stock ticker.',
  backstory="""A seasoned financial analyst with a deep understanding of financial statements, market indicators, and valuation metrics. 
  You provide a clear quantitative picture of a company's financial health.""",
  verbose=True,
  allow_delegation=False,
  tools=[yfinance_tool]
)

# Agent 3: Investment Advisor (Manager)
investment_advisor = Agent(
  role='Lead Investment Advisor',
  goal='Produce a comprehensive investment analysis report by orchestrating a team of specialist agents and verifying their outputs.',
  backstory="""A strategic thinker with a deep understanding of investment principles and market dynamics. 
  You are responsible for the final quality of the investment report, ensuring all data is accurate, relevant, and well-synthesized. 
  You delegate tasks to your team of analysts and review their work critically.""",
  verbose=True,
  allow_delegation=True # IMPORTANT: The manager MUST be allowed to delegate.
)


# --- TASK DEFINITION ---

# We only need one high-level task for the manager agent.
# The manager will break this down into sub-tasks for the other agents.
investment_analysis_task = Task(
  description="""Conduct a comprehensive investment analysis for the company '{company}'. 
  Your final report should be structured with the following sections:
  1.  **Company Profile:** Business summary, sector, industry, and stock ticker.
  2.  **Quantitative Financial Analysis:** Key financial ratios (P/E, P/S, Debt-to-Equity), revenue, EPS, and market cap.
  3.  **Qualitative & Market Analysis:** Summary of recent news and overall market sentiment.
  4.  **Final Synthesis & Recommendation:** A consolidated summary identifying key opportunities and risks, and a final investment outlook.""",
  expected_output="A meticulously formatted and detailed investment report.",
  agent=investment_advisor # Assign the high-level task to the manager.
)

# --- CREW DEFINITION ---

# Instantiate your crew with a hierarchical process
investment_crew = Crew(
  agents=[researcher, financial_analyst, investment_advisor],
  tasks=[investment_analysis_task],
  process=Process.hierarchical,
  manager_llm=None  # We specify the manager LLM here. Defaults to the framework's default LLM if None.
                    # For OpenAI, you can specify a specific model like ChatOpenAI(model_name="gpt-4")
)

# --- EXECUTION ---
if __name__ == "__main__":
    print("## Welcome to the Agentic Investment Analyzer ##")
    print("-----------------------------------------------")
    company_name = input("Please enter the company name you want to analyze: ")
    
    result = investment_crew.kickoff(inputs={'company': company_name})
    
    print("\n\n########################")
    print("## Here is the Final Analysis:")
    print("########################\n")
    print(result)