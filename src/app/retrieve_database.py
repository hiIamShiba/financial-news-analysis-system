import chromadb
import sys

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


def get_context_from_db(user_input):

    ticker = user_input.strip().upper()

    if ticker not in TICKERS_LIST:
        return f"Lỗi: Mã '{ticker}' không nằm trong danh sách hỗ trợ"

    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()

        if not collections:
            return "Lỗi: Không tìm thấy bảng dữ liệu (Collection) nào trong ChromaDB."

        collection = client.get_collection(name=collections[0].name)
        results = collection.get(
            where={"ticker": ticker},
            limit=10  
        )

        documents = results.get('documents', [])

        if not documents:

            query_res = collection.query(
                query_texts=[f"news and financial updates for {ticker}"],
                n_results=5
            )

            documents = query_res.get('documents', [[]])[0]


        if not documents:
            return f"Không tìm thấy dữ liệu tin tức nào cho mã {ticker} trong database."

        context_block = f"### THÔNG TIN CẬP NHẬT CHO MÃ: {ticker} ###\n"

        context_block += (
            "Dưới đây là các tin tức và tóm tắt tài chính "
            "liên quan được truy xuất từ hệ thống:\n\n"
        )

        for i, doc in enumerate(documents):
            context_block += f"Tin {i+1}: {doc}\n"
            context_block += "-" * 30 + "\n"

        return context_block

    except Exception as e:
        return f"Đã xảy ra lỗi khi truy cập Database: {str(e)}"


if __name__ == "__main__":

    print(f"Danh sách mã hỗ trợ: {', '.join(TICKERS_LIST)}")

    user_choice = input(
        "Nhập mã cổ phiếu bạn muốn phân tích: "
    ).upper()

    print("\n--- Đang truy xuất dữ liệu...---")

    final_context = get_context_from_db(user_choice)

    print("\n--- CONTEXT ĐỂ GỬI CHO AI ---\n")

    print(final_context)
