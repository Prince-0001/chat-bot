

import psycopg2
import os
import logging
from groq import Groq  

# Configure logging
logging.basicConfig(level=logging.INFO)

# PostgreSQL connection details
DB_CONFIG = {
    "dbname": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT'),
  
}

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment
groq_client = Groq(api_key=GROQ_API_KEY)

def generate_sql_query(user_input: str) -> str:
    """Uses Groq to generate an SQL query from human input."""
    prompt = f"Convert the following natural language request into an SQL query: {user_input}"
    
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",  # Adjust model as needed
        messages=[{"role": "user", "content": prompt}],
    )
    
    sql_query = response.choices[0].message.content  # Correct way to access the content

    logging.info(f"Generated SQL: {sql_query}")
    
    return sql_query

def execute_sql_query(sql_query: str):
    """Executes the SQL query and returns results."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        logging.error(f"Database error: {e}")
        return f"Error executing query: {e}"

def postgresql_agent(user_input: str):
    """Handles user input related to PostgreSQL and returns query results."""
    sql_query = generate_sql_query(user_input)
    results = execute_sql_query(sql_query)
    return results

# Example usage
if __name__ == "__main__":
    user_query = "Show me all orders placed in the last week."
    response = postgresql_agent(user_query)
    print(response)
