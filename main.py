import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import os
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

df_global = None
monthly_inflow_global = 0
agent = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@app.get("/", response_class=HTMLResponse)
async def main():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/uploadfile/")
async def create_upload_file(csv_file: UploadFile = File(...), monthly_inflow: float = Form(...)):
    global df_global, monthly_inflow_global, agent, memory
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            return {"error": "GOOGLE_API_KEY environment variable not set."}

        df = pd.read_csv(csv_file.file)
        df_global = df
        monthly_inflow_global = monthly_inflow
        
        # Debugging: Print DataFrame head and info
        print("DataFrame Head:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())

        # Basic data validation
        if 'Outflow' not in df.columns or 'Category' not in df.columns or 'Category Group' not in df.columns or 'Date' not in df.columns:
            return {"error": "CSV must have 'Outflow', 'Category', 'Category Group', and 'Date' columns"}

        # Preprocessing
        df['Outflow'] = df['Outflow'].replace({r'\$': ''}, regex=True)
        df['Outflow'] = pd.to_numeric(df['Outflow'], errors='coerce')
        df.dropna(subset=['Outflow'], inplace=True)

        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        # Create a langchain agent
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,
                                            agent_kwargs={
                                                "system_message": "You are a friendly and helpful AI assistant that can analyze spending habits from a CSV file. The CSV file contains 'Outflow', 'Category', 'Category Group', 'Date', 'Month', and 'Year' columns. When asked about dates, use the 'Date', 'Month', and 'Year' columns. Feel free to ask clarifying questions or offer further insights based on the data."
                                            },
                                            memory=memory,
                                            prefix="You are a friendly and helpful AI assistant that can analyze spending habits from a CSV file. The CSV file contains 'Outflow', 'Category', 'Category Group', 'Date', 'Month', and 'Year' columns. When asked about dates, use the 'Date', 'Month', and 'Year' columns. Feel free to ask clarifying questions or offer further insights based on the data.")

        # 50/30/20 Rule Analysis
        needs_spending = df[df['Category Group'] == 'Needs']['Outflow'].sum()
        wants_spending = df[df['Category Group'] == 'Wants']['Outflow'].sum()
        savings_spending = df[df['Category Group'] == 'Savings']['Outflow'].sum()

        total_spending = needs_spending + wants_spending + savings_spending

        needs_percent = (needs_spending / total_spending) * 100
        wants_percent = (wants_spending / total_spending) * 100
        savings_percent = (savings_spending / total_spending) * 100

        recommendations = []

        if needs_percent > 50:
            recommendations.append(f"You are spending {needs_percent:.2f}% on needs, which is over the recommended 50%. Let's look at where you can cut back.")
        if wants_percent > 30:
            recommendations.append(f"You are spending {wants_percent:.2f}% on wants, which is over the recommended 30%. Let's look at where you can cut back.")
        if savings_percent < 20:
            recommendations.append(f"You are saving {savings_percent:.2f}%, which is under the recommended 20%. Let's look for ways to increase your savings.")

        if not recommendations:
            recommendations.append("You are on the right track with the 50/30/20 rule! Great job!")

        return {"recommendations": recommendations}

    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: Request):
    global agent
    data = await request.json()
    message = data.get("message")

    # Debugging: Print user message
    print(f"\nUser Message: {message}")

    if agent is None:
        return {"response": "Please upload a CSV file first."}

    response = agent.invoke({"input": message})["output"]
    
    # Debugging: Print agent response
    print(f"Agent Response: {response}")

    return {"response": response}