import streamlit as st
import pandas as pd
import json
from get_stock_trend import get_stock_historical_trend
from main_agent import run_financial_agent_analysis
from model_manager import SentimentModelManager 

# Page Configuration
st.set_page_config(
    page_title="Agentic AI Financial Analyst",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_model_tool():
    return SentimentModelManager(language="en")

model_tool = get_model_tool()

st.title("🤖 Strategic Agentic AI Financial Analyst")
st.markdown("""
Professional financial analysis system powered by **Llama 3.3 70B** and **FinBERT**. 
This agent performs a multi-step reasoning framework to interpret market events and price action.
""")

tickers_list = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", 
    "META", "TSLA", "BRK-B", "LLY", "V", 
    "JPM", "WMT", "JNJ", "MA", "PG"
]

with st.sidebar:
    st.header("Configuration")
    selected_ticker = st.selectbox("Select Stock Ticker:", tickers_list)
    analyze_btn = st.button("Run Strategic Analysis", type="primary")
    st.divider()
    st.info("**Framework:** 13-Step Chain-of-Thought reasoning.")

if analyze_btn:
    with st.spinner(f"Agent is analyzing {selected_ticker}... Please wait."):
        
        trend_df = get_stock_historical_trend(selected_ticker, days=45)
        result_data = run_financial_agent_analysis(
            selected_ticker,
            model_tool 
        )

        if isinstance(result_data, dict) and "agent_analysis" in result_data:
            report = result_data["agent_analysis"]
            news_list = result_data.get("source_news", [])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Trading Bias", report.get("trading_bias", "N/A").upper())
            with col2:
                st.metric("Bullish Prob.", f"{report.get('bullish_probability')}%")
            with col3:
                st.metric("Confidence Score", f"{report.get('confidence_score')}/100")
            with col4:
                st.metric("Market Regime", report.get("market_regime", "N/A"))

            st.divider()
            st.subheader(f"📈 Price Trend (Last 30 Days): {selected_ticker}")
            if trend_df is not None and not trend_df.empty:
                st.line_chart(trend_df.set_index('Date'))
            else:
                st.warning(f"Trend data for {selected_ticker} is currently unavailable from API.")

            st.divider()
            st.subheader("📰 Data Retrieval & Sentiment Analysis")
            if news_list:
                df_news = pd.DataFrame(news_list)
                df_news.columns = ["#", "News Content", "Sentiment Label", "Confidence Score"]
                st.dataframe(df_news.set_index("#"), width='stretch')
            else:
                st.warning("No recent news found for this ticker in database.")

            st.subheader("🕵️ Strategic Reasoning (Chain-of-Thought)")
            with st.expander("View Full 12-Step Internal Analysis", expanded=True):
                st.write(report.get("detailed_step_by_step_reasoning"))

            c1, c2 = st.columns(2)
            with c1:
                st.success("📈 Bullish Thesis")
                for point in report.get("bullish_thesis", []):
                    st.write(f"• {point}")
            with c2:
                st.error("📉 Bearish Thesis")
                for point in report.get("bearish_thesis", []):
                    st.write(f"• {point}")

            st.warning("⚠️ Identified Risk Factors")
            risks = report.get("major_risks", [])
            st.write(", ".join(risks) if risks else "None identified.")
            
            st.divider()
            st.subheader("🎯 Final Analyst Verdict")
            st.info(report.get("final_reasoning"))
            
        else:
            st.error(f"Analysis Failed. Please check API limits or Database connection.")

st.markdown("---")
st.caption("Disclaimer: Project for Industrial NLP. Not financial advice.")