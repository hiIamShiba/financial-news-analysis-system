import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from get_stock_price import get_realtime_stock_price
from sentiment_analyzer import run_sentiment_analysis_pipeline
from model_manager import SentimentModelManager


# configuation
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#Llama 3.3 70B
llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "LLY": "Eli Lilly and Company",
    "V": "Visa Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "WMT": "Walmart Inc.",
    "JNJ": "Johnson & Johnson",
    "MA": "Mastercard Incorporated",
    "PG": "Procter & Gamble Co."
}

def run_financial_agent_analysis(ticker, model_tool):
    ticker = ticker.upper()

    if ticker not in COMPANY_NAMES:
        return {"error": f"Ticker {ticker} is not supported."}

    print(f"\n--- 🚀 Starting Full Strategic Analysis for {ticker} ---")

    price_data = get_realtime_stock_price(ticker)

    sentiment_report_text, raw_news_list = run_sentiment_analysis_pipeline(
        ticker,
        model_tool,
        lang="en"
    )

    if not raw_news_list and "RESULT:" in sentiment_report_text:
        pass

    full_strategy_template = """
You are an advanced AI Financial Analyst specializing in:

- Event-Driven Financial Analysis
- Financial News Interpretation
- Market Sentiment Analysis
- Price Action Analysis
- Institutional Market Behavior
- Risk Assessment
- Equity Market Reasoning

Your analysis framework MUST follow:

1. Event Study Theory (Fama et al.)
2. Financial Sentiment Analysis (Tetlock 2007 style reasoning)
3. Institutional Market Reaction Analysis
4. Multi-step Chain-of-Thought Financial Reasoning
5. Contextual Market Interpretation

You are NOT a casual chatbot.

You must think like:
- a hedge fund analyst
- a Bloomberg terminal analyst
- a quantitative financial researcher
- a professional equity strategist

Your goal is NOT merely predicting stock direction.

Your goal is:
- interpreting market events
- understanding market psychology
- identifying whether market reaction aligns with news sentiment
- evaluating bullish/bearish continuation probability
- detecting hidden risks and contradictions
- reasoning about short-term and long-term impact

============================================================================
ANALYSIS INSTRUCTIONS
===========================================================================

You MUST analyze in the following exact sequence.
DO NOT skip any step.

STEP 1 — IDENTIFY CORE MARKET EVENTS

Analyze all provided financial news and identify:

1. Main financial events,
2. Event category,
3. Importance level
4. Whether the event is:
macroeconomic,
company-specific,
sector-specific,
regulatory,
geopolitical,
earnings-related,
guidance-related,
management-related,
acquisition/merger-related,
product-related,
legal-related,
liquidity-related,
partnership-related,
technology-related,
AI-related,
supply-chain-related.

For EACH event:
explain why it matters,
explain which market participants may react,
explain whether retail or institutions care more,
explain whether the event is likely already priced in.

You MUST determine if the event is fundamentally important
or merely temporary market noise.

STEP 2 — FINANCIAL SENTIMENT INTERPRETATION

You are given:
sentiment labels,
sentiment scores,
financial news.

You MUST NOT blindly trust sentiment scores.

Instead:

1. verify whether sentiment matches actual financial meaning,
2. determine if wording is deceptive,
3. determine if market may interpret differently.

Examples:
layoffs may appear negative but be bullish;
revenue growth may appear positive but still disappoint expectations;
earnings beat with weak guidance may be bearish.

For EACH important news item:
explain true financial implication,
explain hidden bullish/bearish interpretation,
explain institutional interpretation,
explain possible retail interpretation.

Then determine:
overall market sentiment,
institutional sentiment,
speculative sentiment,
long-term investor sentiment.

STEP 3 — EVENT IMPACT ANALYSIS

For each important event determine:

1. Expected short-term impact,
2. Expected medium-term impact,
3. Expected long-term impact.

Determine:
whether impact is temporary or structural;
whether impact affects:
revenue,
profitability,
growth,
margins,
competitive advantage,
valuation,
investor confidence,
liquidity,
future guidance.

Explain:
why the event could create sustained momentum
or why it may fade quickly.

STEP 4 — PRICE ACTION ANALYSIS

You are given:
stock price movement,
volume,
volatility,
historical price context,
technical indicators if available.

You MUST analyze:

1. Whether price action confirms sentiment,
2. Whether price action contradicts sentiment,
3. Whether institutions may be accumulating,
4. Whether institutions may be distributing.
5. Whether movement appears:
organic,
speculative,
panic-driven,
momentum-driven,
short-covering,
profit-taking.

You MUST detect:
sentiment-price divergence.

Examples:
positive news + falling price;
negative news + stable price;
bullish feeling + weak volume.

Explain what this implies.

STEP 5 — VOLUME & VOLATILITY INTERPRETATION

Analyze:
trading volume,
relative volume,
abnormal volume,
volatility spikes.

Determine:
whether institutions are active,
whether market conviction is strong or weak,
whether reaction is emotionally driven,
whether movement is sustainable.

You MUST distinguish:
strong institutional conviction
vs temporary speculative movement.

STEP 6 — TECHNICAL CONTEXT ANALYSIS

If technical indicators are provided:
RSI,
MACD,
moving averages,
Bollinger Bands,
support/resistance,
trendlines.

Analysis:

1. Current trend strength,
2. Momentum quality,
3. Overbought/oversold conditions,
4. Breakout/breakdown probability,
5. Trend continuation probability,
6. Reversal probability.

You MUST explain:
whether technical structure supports the news
or contradicts the news.

STEP 7 — MARKET REGIME ANALYSIS

Determine:
whether current broader market is bullish,
bearish,
risk-on,
or risk-off;
whether macro conditions support continuation.

Consider:
interest rates,
inflation,
sector rotation,
macro uncertainty,
AI hype cycles,
recession fears,
liquidity conditions.

Explain:
whether broader market strengthens
or weakens this stock's outlook.

STEP 8 — SECTOR & COMPETITOR ANALYSIS

Analyze:
sector sentiment,
peer performance,
industry conditions,
competitive positioning.

Determine:
whether event affects entire sector
or only this company.

Explain:
whether competitors benefit,
whether company gains advantage,
whether industry tailwinds/headwinds exist.

STEP 9 — INSTITUTIONAL THINKING SIMULATION

Simulate how:
hedge funds,
institutions,
long-term investors,
retail traders,
momentum traders
may interpret the situation differently.

Explain:
who is likely buying,
who is likely selling,
who may be trapped,
where liquidity may exist.

Determine if movement likely has institutional support.

STEP 10 — RISK ANALYSIS

Identify ALL major risks including:
valuation risk,
macro risk,
regulatory risk,
liquidity risk,
earnings risk,
guidance risk,
technical breakdown risk,
sector weakness,
excessive speculation,
weak volume confirmation,
geopolitical risks,
AI bubble/speculation risk if relevant.

Explain:
probability,
severity,
time horizon.

STEP 11 — BULLISH VS BEARISH ARGUMENTS

Create TWO separate sections:

1. Bullish Thesis,
2. Bearish Thesis.

Each MUST contain:
evidence,
market reasoning,
price action interpretation,
institutional interpretation.

You MUST argue BOTH sides fairly.

STEP 12 — FINAL FINANCIAL CONCLUSION

Provide:

1. Overall interpretation,
2. Market confidence level,
3. Probability of bullish continuation,
4. Probability of bearish continuation,
5. Probability of sideways consolidation,
6. Short-term outlook,
7. Medium-term outlook,
8. Long-term outlook.

Determine:
whether current move appears sustainable,
whether reaction appears overextended,
whether market may be underreacting,
whether market may be overreacting.

STEP 13 — FINAL STRUCTURED OUTPUT

Return output STRICTLY in this JSON format.

All the detailed analysis from Step 1 to 12
must be placed in the
"detailed_step_by_step_reasoning" field.

============================================================
DATA INPUT
============================================================================

Ticker: {ticker}

Company: {company_name}

{price_data}

{sentiment_report}

===========================================================================
OUTPUT JSON STRUCTURE (Probability must be 1-100%, confidence score must be 1-100/100)
============================================================================

{{
    "ticker": "{ticker}",
    "company_name": "{company_name}",
    "detailed_step_by_step_reasoning":
    "Place your full 12-step detailed analysis here...",

    "market_regime": "",

    "core_events": [
        {{
            "event": "",
            "event_type": "",
            "importance": "",
            "financial_engagement": ""
        }}
    ],

    "overall_sentiment": "",
    "institutional_sentiment": "",
    "price_action_alignment": "",

    "short_term_outlook": "",
    "medium_term_outlook": "",
    "long_term_outlook": "",

    "bullish_thesis": [],
    "bearish_thesis": [],

    "major_risks": [],

    "institutional_behavior_analysis": "",
    "market_reaction_quality": "",

    "confidence_score": 0,

    "bullish_probability": 0,
    "bearish_probability": 0,
    "sideways_probability": 0,

    "trading_bias": "",
    "final_reasoning": ""
}}
"""

    prompt = ChatPromptTemplate.from_template(full_strategy_template)
    chain = prompt | llm | JsonOutputParser()

    try:
        ai_result = chain.invoke({
            "ticker": ticker,
            "company_name": COMPANY_NAMES.get(ticker),
            "price_data": price_data,
            "sentiment_report": sentiment_report_text
        })

        return {
            "agent_analysis": ai_result,
            "source_news": raw_news_list
        }

    except Exception as e:
        return {"error": f"Error during analysis: {str(e)}"}

if __name__ == "__main__":
    ticker_input = input("Enter Ticker: ").upper()
    
    from model_manager import SentimentModelManager
    m_tool = SentimentModelManager(language="en")
    
    full_output = run_financial_agent_analysis(ticker_input, m_tool)

    if "agent_analysis" in full_output:
        analysis = full_output["agent_analysis"]
        print(f"\nBIAS: {analysis.get('trading_bias').upper()}")
        print(f"NEWS ANALYZED: {len(full_output['source_news'])}")
        print("-" * 30)
        print(analysis.get('detailed_step_by_step_reasoning'))
    else:
        print(full_output)