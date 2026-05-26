import finnhub
import os
from datetime import datetime
from dotenv import load_dotenv


# 1. Supported Tickers List
TICKERS_LIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "BRK-B",
    "LLY",
    "V",
    "JPM",
    "WMT",
    "JNJ",
    "MA",
    "PG"
]


def get_realtime_stock_price(ticker_input):
    """
    Fetches real-time stock data from Finnhub API.
    Returns a formatted English string for LLM processing.
    """

    ticker = ticker_input.strip().upper()

    if ticker not in TICKERS_LIST:
        return f"ERROR: {ticker} is not in supported tickers list."


    load_dotenv()
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

    finnhub_client = finnhub.Client(
        api_key=FINNHUB_API_KEY
    )

    try:


        quote = finnhub_client.quote(ticker)

        if quote is None or quote['c'] == 0:
            return (
                f"DATA_ERROR: Could not retrieve price for "
                f"{ticker}. The market might be closed "
                f"or API limit reached."
            )


        price_change = quote['d']
        percent_change = quote['dp']

        trend = (
            "BULLISH/UP"
            if price_change > 0
            else "BEARISH/DOWN"
            if price_change < 0
            else "NEUTRAL/FLAT"
        )

        price_block = (
            f"### REAL-TIME MARKET DATA: {ticker} ###\n"
        )

        price_block += (
            f"- Retrieval Timestamp: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"(UTC)\n"
        )

        price_block += (
            f"- Current Price: ${quote['c']}\n"
        )

        price_block += (
            f"- Market Trend: {trend}\n"
        )

        price_block += (
            f"- Price Change: "
            f"{'+' if price_change > 0 else ''}"
            f"{price_change} "
            f"({percent_change}%)\n"
        )

        price_block += (
            f"- Day High: ${quote['h']}\n"
        )

        price_block += (
            f"- Day Low: ${quote['l']}\n"
        )

        price_block += (
            f"- Open Price: ${quote['o']}\n"
        )

        price_block += (
            f"- Previous Close: ${quote['pc']}\n"
        )

        return price_block

    except Exception as e:
        return (
            f"SYSTEM_ERROR: Finnhub API failed "
            f"with message: {str(e)}"
        )



if __name__ == "__main__":

    print(
        f"Supported Tickers: "
        f"{', '.join(TICKERS_LIST)}"
    )

    user_choice = input(
        "Enter Ticker to fetch: "
    ).upper()

    print("\n--- Fetching market data... ---")

    stock_info = get_realtime_stock_price(
        user_choice
    )

    print("\n[OUTPUT FOR AGENT]:")

    print(stock_info)