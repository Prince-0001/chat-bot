import psycopg2
import os
import logging
from groq import Groq  

# Configure logging
logging.basicConfig(level=logging.INFO)

# PostgreSQL connection details (UPDATE THESE)
DB_CONFIG = {
    "dbname": "mydatabase",
    "user": "admin",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}

# Groq API Key (set it as an environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("\u274c ERROR: GROQ_API_KEY is not set! Please configure it.")

groq_client = Groq(api_key=GROQ_API_KEY)

# Few-Shot Example for LLM
FEW_SHOT_EXAMPLES = """
Your task is to generate SQL queries for a PostgreSQL database using the schema provided. DO NOT include explanations, just return the SQL query.

Database Schema:
- `products` table: (`id`, `name`, `price`, `category`)
- `users` table: (`id`, `name`, `email`, `created_at`)
- `orders` table: (`id`, `order_date`, `total_amount`, `userId`)

Examples:
1️⃣ User Input: "Get the price of a T-Shirt"
   SQL: ```sql
   SELECT price FROM products WHERE name = 'T-Shirt';
   ```

2️⃣ User Input: "Find users who signed up in the last 30 days"
   SQL: ```sql
   SELECT * FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';
   ```

3️⃣ User Input: "Show all orders placed by user with ID 5"
   SQL: ```sql
   SELECT * FROM orders WHERE userId = 5;
   ```

4️⃣ User Input: "Get total sales amount for the last 7 days"
   SQL: ```sql
   SELECT SUM(total_amount) FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '7 days';
   ```

Now, generate a SQL query for the following input:
"""

def clean_sql(query: str) -> str:
    """Removes triple backticks from Groq's SQL response."""
    return query.replace("```sql", "").replace("```", "").strip()

def generate_sql_query(user_input: str) -> str:
    """Uses Groq to generate an SQL query from natural language using a few-shot prompt."""
    prompt = FEW_SHOT_EXAMPLES + f"\nUser Input: \"{user_input}\"\nSQL:"

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_sql = response.choices[0].message.content
        sql_query = clean_sql(raw_sql)  # ✅ Fix: Remove ```sql ... ``` formatting

        logging.info(f"✅ Generated SQL Query:\n{sql_query}")
        return sql_query

    except Exception as e:
        logging.error(f"❌ Error generating SQL: {e}")
        return None

def execute_sql_query(sql_query: str):
    """Executes the SQL query and returns results."""
    if not sql_query:
        return "❌ No valid SQL query to execute."

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result

    except psycopg2.OperationalError as e:
        logging.error(f"❌ Database connection error: {e}")
        return "❌ Failed to connect to the database. Check DB settings."
    except Exception as e:
        logging.error(f"❌ Database query error: {e}")
        return f"❌ Error executing query: {e}"

def postgresql_agent(user_input: str):
    """Handles user input related to PostgreSQL and returns query results."""
    sql_query = generate_sql_query(user_input)
    results = execute_sql_query(sql_query)
    return results

# 🚀 Example Usage
if __name__ == "__main__":
    user_query = input("🔹 Enter your query: ")
    response = postgresql_agent(user_query)
    print("🔹 Query Result:", response)
