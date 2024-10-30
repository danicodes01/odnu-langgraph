import os
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

NASA_API_KEY = os.getenv("NASA_API_KEY")

NASA_APOD_URL = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"

def get_nasa_apod():
    """Fetches the NASA Astronomy Picture of the Day (APOD) and its description.
    
    Returns:
        dict: A dictionary containing the title, date, explanation, and image URL of the APOD.
    """
    response = requests.get(NASA_APOD_URL)
    response.raise_for_status()  # Raise an error if the request fails
    data = response.json()
    return {
        "title": data.get("title", "N/A"),
        "date": data.get("date", "N/A"),
        "explanation": data.get("explanation", "N/A"),
        "image_url": data.get("url", "N/A")
    }

def add(a: int, b: int) -> int:
    """Adds a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a / b

# Combine all tools
tools = [
    TavilySearchResults(max_results=1),
    add,
    multiply,
    divide,
    get_nasa_apod
]