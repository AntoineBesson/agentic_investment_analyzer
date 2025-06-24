# In tools/financial_tools.py
import yfinance as yf
from langchain_core.tools import BaseTool

class YFinanceTool(BaseTool):
    name: str = "Yahoo Finance Tool"
    description: str = "A tool to get financial data from Yahoo Finance for a given stock ticker. It can fetch company info, key ratios, and financial statements."

    def _get_company_info(self, ticker: str) -> dict:
        """Helper function to get company information."""
        return yf.Ticker(ticker).info

    def _run(self, ticker: str) -> str:
        """
        The main execution method for the tool.
        For this implementation, we will fetch and format key company info.
        You should expand this to return financial statements, ratios, etc.
        """
        try:
            print(f"--- Using YFinanceTool for ticker: {ticker} ---")
            info = self._get_company_info(ticker)
            
            # Select and format key information for the agent
            formatted_info = {
                "Company Name": info.get('longName'),
                "Ticker": ticker,
                "Sector": info.get('sector'),
                "Industry": info.get('industry'),
                "Business Summary": info.get('longBusinessSummary'),
                "Market Cap": f"${info.get('marketCap', 0):,}",
                "Price-to-Earnings (P/E) Ratio": info.get('trailingPE'),
                "Price-to-Sales (P/S) Ratio": info.get('priceToSalesTrailing12Months'),
                "Debt-to-Equity Ratio": info.get('debtToEquity')
            }
            return str(formatted_info) # Return as a string for the LLM
        except Exception as e:
            return f"An error occurred while fetching data for ticker {ticker}: {e}"