import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)

# Xác định đường dẫn thư mục gốc (root) của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_JSON = os.path.join(BASE_DIR, "data", "processed", "news_with_sentiment.json")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for news_id, item in data.items():
        # 1. Giải quyết vấn đề metadata: Chuyển list thành string
        tickers_str = ", ".join(item.get("affected_tickers", []))

        # 2. Xây dựng nội dung chính để vectorize (Searchable text)
        page_content = f"{item['title']}\n{item['summary']}"

        # 3. Tạo metadata dict (chỉ chứa các kiểu dữ liệu nguyên thủy)
        metadata = {
            "news_id": news_id,
            "title": item["title"],
            "pubDate": item["pubDate"],
            "publisher": item["publisher"],
            "link": item["link"],
            "affected_tickers": tickers_str,
            "sentiment": item["sentiment"],
            "sentiment_score": item["sentiment_score"],
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Đã chuẩn bị {len(documents)} documents. Khởi tạo embedding model...")

    # Sử dụng model embedding chuyên dụng cho vector search
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

    print("Khởi tạo ChromaDB và bắt đầu ingest theo từng batch...")

    # Khởi tạo đối tượng Chroma (chưa đưa dữ liệu vào vội)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="financial_news",
    )

    # Cấu hình batch_size an toàn (nhỏ hơn 5461)
    BATCH_SIZE = 5000

    # Vòng lặp cắt danh sách documents và add dần vào DB
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"Đang đẩy batch từ {i} đến {i + len(batch)} / {len(documents)}...")
        vector_store.add_documents(documents=batch)

    # Lưu lại thay đổi xuống ổ cứng
    vector_store.persist()
    print("Ingestion hoàn tất.")


if __name__ == "__main__":
    main()
