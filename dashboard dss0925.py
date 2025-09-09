import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- Load Environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# --- Constants ---
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)


# ===================== UTILS ===================== #
def fetch_data(endpoint: str, params: dict = None):
    """Generic function to fetch data from Sectors API."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def run_llm(prompt_template: str, data: pd.DataFrame):
    """Format prompt with data and invoke LLM."""
    prompt = PromptTemplate.from_template(prompt_template).format(data=data.to_string(index=False))
    return llm.invoke(prompt).content


def clean_python_code(raw_code: str):
    """Cleans LLM-generated Python code block."""
    return raw_code.strip().strip("```").replace("python", "").strip()


# ===================== SECTIONS ===================== #
def sidebar_selector():

    """Sidebar for subsector & company"""

    st.sidebar.title("📌 Analytic Selection")

    subsectors = fetch_data("subsectors/")
    subsector_list = pd.DataFrame(subsectors)["subsector"].sort_values().tolist()

    ## streamlit UI
    selected_subsector = st.sidebar.selectbox("🔽 Select Subsector", subsector_list)

    companies = fetch_data("companies/", params={"sub_sector": selected_subsector})
    companies_df = pd.DataFrame(companies)
    company_options = companies_df["symbol"] + " - " + companies_df["company_name"]

    ## streamlit UI
    selected_company = st.sidebar.selectbox("🏢 Selec Company", company_options)

    return selected_company.split(" - ")[0]  # return symbol


def financial_summary(symbol: str):

    """Financial Executive Summary from an LLM."""

    financials = pd.DataFrame(fetch_data(f"financials/quarterly/{symbol}/",
                                         params={"n_quarters": "4",
                                                "report_date": "2023-09-30"}))

    prompt = """
        You are a skilled financial analyst.
        Based on the following quarterly financial data:

        {data}

        Write a three-point executive summary for an investor.
        Focus on:
        1. Revenue growth trends
        2. Profitability
        3. Operating Cash Flow Position
    """
    summary = run_llm(prompt, financials)

    with st.expander("💡 Financial Summary"):
        st.markdown(summary)

    return financials


def revenue_trend(symbol: str, financials: pd.DataFrame):

    """Generate line plot for Revenue Trend."""

    data_sample = financials[['date', 'revenue']].dropna()

    prompt = f"""
        As an expert Python programmer specializing in data visualization.

        Here is the company's revenue data:

        {data}

        Create a Python script using matplotlib to generate a line plot.

        Instructions:
        The X-axis should be 'date'.
        The Y-axis should be 'revenue'.
    
        Write ONLY the executable Python code. Do not include any explanations.
    """
    code = clean_python_code(llm.invoke(prompt).content)

    with st.expander("📊 Visualisasi Tren Pendapatan"):
        exec_locals = {}
        exec(code, {}, exec_locals)
        st.pyplot(exec_locals["fig"])


def trend_analysis(financials: pd.DataFrame):
    """Financial trend interpretation using an LLM."""
    prompt ="""
        Act as a financial analyst.
        Based on the following quarterly data:
        {data}
        Analyze the main trends emerging from the data. Focus on the movement of revenue, net income, and operating cash flow.
        Provide the analysis in 3 concise points.
    """

    analysis = run_llm(prompt, financials)
    with st.expander("🔎 Interpretasi Tren Keuangan"):
        st.markdown(analysis)


def risk_analysis(financials: pd.DataFrame):

    """Financial Risk Analysis with an LLM"""

    prompt = """
        Act as a skeptical financial risk analyst.
        Carefully examine the following financial data:
        {data}
        Identify 2-3 potential risks or "red flags" that should be a cause for concern. For each point, provide a brief, one-sentence explanation.
    """
    risks = run_llm(prompt, financials)
    with st.expander("⚠️ Potensi Risiko Keuangan"):
        st.markdown(risks)


# ===================== MAIN APP ===================== #
def main():
    symbol = sidebar_selector()

    if st.sidebar.button("🔍 Lihat Insight"):
        financials = financial_summary(symbol)
        revenue_trend(symbol, financials)
        trend_analysis(financials)
        risk_analysis(financials)


if __name__ == "__main__":
    main()
