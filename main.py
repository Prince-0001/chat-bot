import logging
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from agents.rag_agent import retrieve_rag_response
from agents.search_agent import searchAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function: Supervisor Agent (routes queries)
def supervisorAgent(state):
    query = state["query"].lower()
    logging.info(f"Processing query: {query}")

    if any(word in query for word in ["latest", "current", "top", "news", "trending"]):
        logging.info("Routing to Search Agent.")
        return {"next_step": "search_agent", "query": query}

    elif any(word in query for word in ["define", "explain", "history", "background"]):
        logging.info("Routing to RAG Agent.")
        return {"next_step": "rag_agent", "query": query}

    elif any(word in query for word in ["count", "total", "number", "list", "fetch"]):
        logging.info("Routing to SQL Agent.")
        return {"next_step": "sql_agent", "query": query}

    else:
        logging.info("Defaulting to RAG Agent.")
        return {"next_step": "rag_agent", "query": query}

# Function: SQL Agent
def sql_agent(state):
    logging.info(f"SQL Agent processing query: {state['query']}")
    return {"response": f"SQL Agent: Retrieved structured data for '{state['query']}'."}

# Function: Web Search Agent (DuckDuckGo)
def search_agent(state):
    logging.info(f"Search Agent processing query: {state['query']}")
    search_results = searchAgent(state)  # Ensure this returns a dict with "response"
    return {"response": search_results.get("response", "No search results found.")}

# Function: RAG Agent (Retrieves from vector DB)
def rag_agent(state):  
    query = state["query"]
    logging.info(f"RAG Agent processing query: {query}")
    
    try:
        docs = retrieve_rag_response(query)
        if isinstance(docs, list):  
            response = "\n".join([doc.page_content for doc in docs[:3]])  # Ensure formatting
        else:
            response = docs

    except Exception as e:
        logging.error(f"RAG Agent Error: {e}")
        response = "RAG Agent encountered an error."
    
    return {"response": response}

# Define the workflow graph
workflow = StateGraph(dict)

workflow.add_node("supervisor", supervisorAgent)  # Updated supervisor function
workflow.add_node("sql_agent", sql_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("rag_agent", rag_agent)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next_step"])
workflow.add_edge("sql_agent", END)
workflow.add_edge("search_agent", END)
workflow.add_edge("rag_agent", END)

# Compile workflow graph
graph = workflow.compile()

# Interactive CLI
print(" Welcome to the Agent Query Routing System! Type 'exit' to quit.\n")
while True:
    user_query = input(" Ask your question: ")
    
    if user_query.lower() == "exit":
        print("Exiting the system. Goodbye!")
        break
    
    response = graph.invoke({"query": user_query})
    print(" AI:", response["response"], "\n")
