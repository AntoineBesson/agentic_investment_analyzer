# In tools/scraping_tools.py
import requests
from bs4 import BeautifulSoup
from crewai_tools import BaseTool

class ScrapeWebsiteTool(BaseTool):
    name: str = "Scrape Website Tool"
    description: str = "Scrapes the content of a given URL. Useful for reading articles and company descriptions."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from the body, you can refine this selector
            text_content = soup.body.get_text(separator=' ', strip=True)
            return text_content[:4000] # Return the first 4000 characters to avoid huge context
        except requests.exceptions.RequestException as e:
            return f"Error accessing URL: {e}"
        except Exception as e:
            return f"An error occurred during scraping: {e}"