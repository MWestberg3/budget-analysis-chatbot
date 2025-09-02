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

def calculate_50_30_recommendations(df: pd.DataFrame, monthly_inflow: float):
    needs_spending = df[df['Category Group'] == 'Needs']['Outflow'].sum()
    wants_spending = df[df['Category Group'] == 'Wants']['Outflow'].sum()
    
    total_net_spending = df['Outflow'].sum() - df['Inflow'].sum()

    recommendations = []

    if monthly_inflow > 0: # Calculate percentages based on monthly inflow
        needs_percent = (needs_spending / monthly_inflow) * 100
        wants_percent = (wants_spending / monthly_inflow) * 100

        # Debugging: Print calculated percentages
        print(f"\nNeeds Spending: {needs_spending:.2f}")
        print(f"Wants Spending: {wants_spending:.2f}")
        print(f"Total Net Spending: {total_net_spending:.2f}")
        print(f"Needs Percent (vs Inflow): {needs_percent:.2f}%")
        print(f"Wants Percent (vs Inflow): {wants_percent:.2f}%")

        if needs_percent > 50:
            recommendations.append(f"Your spending on Needs is {needs_percent:.2f}% of your inflow, which is over the recommended 50%. Consider reviewing these essential expenses.")
        if wants_percent > 30:
            recommendations.append(f"Your spending on Wants is {wants_percent:.2f}% of your inflow, which is over the recommended 30%. Look for areas to cut back on discretionary spending.")
        
        if needs_percent <= 50 and wants_percent <= 30:
            recommendations.append("Great job! Your spending on Needs and Wants is within the recommended 50/30 rule relative to your inflow.")
    else:
        recommendations.append("Monthly inflow not provided, cannot assess spending against 50/30 rule.")

    # Add recommendation based on total net spending vs monthly inflow
    if monthly_inflow > 0:
        if total_net_spending > monthly_inflow:
            recommendations.append(f"Your total net spending (${total_net_spending:.2f}) exceeds your monthly inflow (${monthly_inflow:.2f}). You are spending more than you earn.")
        else:
            savings = monthly_inflow - total_net_spending
            recommendations.append(f"You have a surplus of ${savings:.2f} this month. Consider saving or investing this amount.")
    else:
        recommendations.append("Monthly inflow not provided, cannot assess spending against income.")

    return recommendations

def process_spending_data(df: pd.DataFrame, monthly_inflow: float):
    # Preprocessing
    # Remove all non-digit characters except for the decimal point
    df['Outflow'] = df['Outflow'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Outflow'] = pd.to_numeric(df['Outflow'], errors='coerce')
    df['Inflow'] = df['Inflow'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Inflow'] = pd.to_numeric(df['Inflow'], errors='coerce')
    df.dropna(subset=['Outflow', 'Inflow'], inplace=True)

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Group spending by month and compare to monthly inflow
    monthly_summary = []
    for (year, month), group in df.groupby(['Year', 'Month']):
        month_name = pd.to_datetime(str(month), format='%m').strftime('%B')
        total_net_spent_month = group['Outflow'].sum() - group['Inflow'].sum()
        needs_spent_month = group[group['Category Group'] == 'Needs']['Outflow'].sum()
        wants_spent_month = group[group['Category Group'] == 'Wants']['Outflow'].sum()
        
        # Calculate other_spent_month explicitly
        other_spent_month = group[~group['Category Group'].isin(['Needs', 'Wants'])]['Outflow'].sum()

        month_summary = {
            "month": f"{month_name} {year}",
            "total_net_spent": total_net_spent_month,
            "needs_spent": needs_spent_month,
            "wants_spent": wants_spent_month,
            "other_spent": other_spent_month,
            "recommendations": []
        }

        # Compare to monthly inflow
        if monthly_inflow > 0:
            if total_net_spent_month > monthly_inflow:
                month_summary["recommendations"].append(f"Warning: In {month_name} {year}, your net spending (${total_net_spent_month:.2f}) exceeded your monthly inflow (${monthly_inflow:.2f}).")
            else:
                month_summary["recommendations"].append(f"Info: In {month_name} {year}, your net spending (${total_net_spent_month:.2f}) was within your monthly inflow (${monthly_inflow:.2f}).")
        else:
            month_summary["recommendations"].append(f"Info: Monthly inflow not provided, cannot compare spending for {month_name} {year}.")

        # Compare to 50/30 rule for the month
        if monthly_inflow > 0:
            needs_percent_month = (needs_spent_month / monthly_inflow) * 100
            wants_percent_month = (wants_spent_month / monthly_inflow) * 100

            if needs_percent_month > 50:
                month_summary["recommendations"].append(f"Warning: In {month_name} {year}, your Needs spending was {needs_percent_month:.2f}% of your inflow, exceeding the 50% guideline.")
            if wants_percent_month > 30:
                month_summary["recommendations"].append(f"Warning: In {month_name} {year}, your Wants spending was {wants_percent_month:.2f}% of your inflow, exceeding the 30% guideline.")
            
            if needs_percent_month <= 50 and wants_percent_month <= 30:
                month_summary["recommendations"].append(f"Info: In {month_name} {year}, your Needs and Wants spending adhered to the 50/30 rule relative to your inflow.")
        else:
            month_summary["recommendations"].append(f"Info: Monthly inflow not provided, cannot assess 50/30 rule for {month_name} {year}.")

        monthly_summary.append(month_summary)
    return monthly_summary

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
        if 'Outflow' not in df.columns or 'Inflow' not in df.columns or 'Category' not in df.columns or 'Category Group' not in df.columns or 'Date' not in df.columns:
            return {"error": "CSV must have 'Outflow', 'Inflow', 'Category', 'Category Group', and 'Date' columns"}

        # Call the new function to process data and get monthly summary
        monthly_summary = process_spending_data(df, monthly_inflow_global)

        # Create a langchain agent
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,
                                            agent_kwargs={
                                                "system_message": "You are a friendly and helpful AI assistant that can answer questions about spending habits from a CSV file. The CSV file contains 'Outflow', 'Category', 'Category Group', 'Date', 'Month', and 'Year' columns. When asked about dates, use the 'Date', 'Month', and 'Year' columns. Feel free to ask clarifying questions or offer further insights based on the data."
                                            },
                                            memory=memory,
                                            prefix="You are a friendly and helpful AI assistant that can analyze spending habits from a CSV file. The CSV file contains 'Outflow', 'Category', 'Category Group', 'Date', 'Month', and 'Year' columns. When asked about dates, use the 'Date', 'Month', and 'Year' columns. Feel free to ask clarifying questions or offer further insights based on the data.")

        # Overall recommendations (from calculate_50_30_recommendations)
        overall_recommendations = calculate_50_30_recommendations(df, monthly_inflow_global)

        # Combine monthly and overall recommendations
        all_recommendations = []
        for summary in monthly_summary:
            all_recommendations.append(f"\n--- {summary['month']} ---")
            all_recommendations.append(f"Total Net Spent: ${summary['total_net_spent']:.2f}")
            all_recommendations.append(f"Needs Spent: ${summary['needs_spent']:.2f}")
            all_recommendations.append(f"Wants Spent: ${summary['wants_spent']:.2f}")
            all_recommendations.append(f"Other Spent: ${summary['other_spent']:.2f}")
            all_recommendations.extend(summary['recommendations'])
        
        all_recommendations.append("\n--- Overall Recommendations ---")
        all_recommendations.extend(overall_recommendations)

        return {"recommendations": all_recommendations}

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