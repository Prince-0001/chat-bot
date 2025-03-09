
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_community.tools import DuckDuckGoSearchRun

import os

# Load environment variables
load_dotenv()

model = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")
)
response=model.invoke("2 multiply 3")
print(response.content)

def search_duckduckgo(query:str):
    """Searches DuckDuckgo using Langchain's  DuckDuckGoSearchRun tool."""
    search=DuckDuckGoSearchRun()
    return search.invoke(query)

def add(a:float, b:float)-> float:
    """Add two numbers"""
    return a+b

def multiply (a:float, b:float)-> float:
    """Multiply tow numbers."""
    return a*b

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt=(
        "You are a math expert. Always use one tool at a time. "
        "If the input is a multiplication problem, use the multiply function. "
        "If the input is an addition problem, use the add function."
    )
)

research_agent = create_react_agent(
    model=model,
    tools=[search_duckduckgo],
    name="research_expert",
    prompt="You are a world-class researcher with access to web search. Do not do any math."
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

app = workflow.compile()

# Change the user input to be more explicit
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "weather of india?"
        }
    ]
})

# Debugging: Print the result to see what is returned


for m in result["messages"]:
    print(m)