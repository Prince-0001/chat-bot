import logging
from langchain_community.tools import DuckDuckGoSearchRun

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def searchAgent(state):
    query = state["query"]
    logging.info(f"Search Agent received query: {query}")

    search = DuckDuckGoSearchRun()
    try:
        results = search.invoke(query)
        # logging.info(f"Raw search results: {results}")

        # If results are a list of dicts (expected format)
        if isinstance(results, list) and all(isinstance(res, dict) for res in results):
            response = "\n".join(
                [f"- {res.get('title', 'No Title')}: {res.get('link', 'No Link')}" for res in results[:3]]
            )

        # If results are a string (fallback case)
        elif isinstance(results, str):
            
            response = results[:500]  # Limit length to avoid excessive output

        else:
            logging.warning("Search Agent received no structured results.")
            response = "No relevant search results found."

    except Exception as e:
        logging.error(f"Search Agent Error: {e}", exc_info=True)
        response = "Search Agent encountered an error."

    return {"response": response}
