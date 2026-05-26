import chromadb
from model_manager import SentimentModelManager

TICKERS_LIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", 
    "META", "TSLA", "BRK-B", "LLY", "V", 
    "JPM", "WMT", "JNJ", "MA", "PG"
]

CHROMA_PATH = "./chroma_db"

def run_sentiment_analysis_pipeline(
    ticker_input,
    model_tool,
    lang="en"
):
    """
    Sentiment Analysis Pipeline.
    
    Returns:
        tuple: (analysis_block_string, list_of_news_details)
    """

    ticker = ticker_input.strip().upper()

    if ticker not in TICKERS_LIST:
        return f"Ticker {ticker} is not in the supported list.", []

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()

        if not collections:
            return "ChromaDB database is currently empty.", []

        collection = client.get_collection(
            name=collections[0].name
        )

        print(f"--- Searching news for {ticker}... ---")

        db_res = collection.get(
            where={"ticker": ticker},
            limit=10
        )

        docs = db_res.get('documents', [])

        if not docs:
            print(f"⚠️ No metadata 'ticker'={ticker} found. Trying Semantic Search...")
            # Step 2: Semantic Search
            query_res = collection.query(
                query_texts=[
                    f"Latest financial news and stock market updates for {ticker}"
                ],
                n_results=10
            )
            docs = query_res.get('documents', [[]])[0]

        if not docs:
            return f"RESULT: No news found for {ticker} in the database.", []

        print(f"✅ Found {len(docs)} articles. Analyzing sentiment...")
        sentiment_results = model_tool.get_sentiment(docs)

        analysis_block = f"\n=== MARKET SENTIMENT REPORT: {ticker} ===\n"
        analysis_block += f"Total articles analyzed: {len(docs)}\n"
        analysis_block += "-" * 40 + "\n"
        
        ui_news_data = [] 

        for i, (doc, sent) in enumerate(zip(docs, sentiment_results)):
            label = sent['label']
            score = round(sent['score'], 4)

            ui_news_data.append({
                "index": i + 1,
                "content": doc,
                "sentiment": label.upper(),
                "confidence": score
            })

            short_doc = doc[:200] + "..." if len(doc) > 200 else doc
            analysis_block += f"News {i+1}: {short_doc}\n"
            analysis_block += f">> Sentiment Label: {label.upper()} (Confidence: {score})\n"
            analysis_block += "-" * 30 + "\n"

        return analysis_block, ui_news_data

    except Exception as e:
        return f"System Error during retrieval: {str(e)}", []


if __name__ == "__main__":
    print(f"SUPPORTED TICKERS: {', '.join(TICKERS_LIST)}")

    ticker_choice = input("\nEnter Ticker to test: ").upper()
    
    print("--- Initializing Model for testing... ---")
    test_model_tool = SentimentModelManager(language="en")

    agent_text, ui_data = run_sentiment_analysis_pipeline(
        ticker_choice,
        test_model_tool
    )

    print("\n" + "=" * 50)
    print("[FINAL OUTPUT FOR AGENT]")
    print(agent_text)
    
    print("\n[EXTRACTED DATA FOR UI]")
    for item in ui_data:
        print(f"Item {item['index']}: {item['sentiment']} ({item['confidence']})")
    print("=" * 50)