import finnhub
import json
import os
import time
from datetime import datetime, timedelta

# Khởi tạo client Finnhub (cần thay thế bằng API key thực tế)
finnhub_client = finnhub.Client(api_key="d7nsl9hr01qs975tc1t0d7nsl9hr01qs975tc1tg")

# Khởi tạo danh sách các mã chứng khoán cần lấy tin tức
tickers_list = [
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
    "PG",
]

# Thiết lập khoảng thời gian lấy tin (Finnhub bắt buộc _from và to)
date_from_str = "2026-01-01"
date_to_str = "2026-04-27"

start_date_global = datetime.strptime(date_from_str, "%Y-%m-%d")
end_date_global = datetime.strptime(date_to_str, "%Y-%m-%d")

# Đường dẫn file database cục bộ
db_file = "../../data/raw/bert_news_db.json"

# Tải dữ liệu cũ nếu file đã tồn tại
if os.path.exists(db_file):
    with open(db_file, "r", encoding="utf-8") as f:
        unique_articles = json.load(f)
else:
    # Sử dụng dictionary để lưu trữ và loại bỏ trùng lặp dựa trên ID bài báo
    unique_articles = {}
    os.makedirs(os.path.dirname(db_file), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

for ticker_symbol in tickers_list:
    current_start_date = start_date_global

    while current_start_date <= end_date_global:
        # Cộng thêm 6 ngày để tạo thành cửa sổ 1 tuần (tính cả ngày bắt đầu là 7 ngày)
        current_end_date = current_start_date + timedelta(days=6)

        # Đảm bảo không vượt quá ngày kết thúc tổng
        if current_end_date > end_date_global:
            current_end_date = end_date_global

        str_from = current_start_date.strftime("%Y-%m-%d")
        str_to = current_end_date.strftime("%Y-%m-%d")

        # Lấy danh sách tin tức liên quan đến mã chứng khoán từ Finnhub
        news_data = finnhub_client.company_news(
            ticker_symbol, _from=str_from, to=str_to
        )

        # Tạm dừng để tránh lỗi 429 Too Many Requests từ Finnhub
        time.sleep(1)

        # In thông tin các bài báo
        for article in news_data:
            # In toàn bộ từ điển để xác định các khóa (keys) thực tế hiện tại
            # print(article)

            # Lấy ID duy nhất của bài báo để kiểm tra trùng lặp
            article_id = str(article.get("id"))

            if not article_id or article_id == "None":
                continue

            # Kiểm tra xem bài báo đã tồn tại trong dictionary chưa
            if article_id in unique_articles:
                # Nếu đã tồn tại, kiểm tra và thêm Ticker hiện tại vào danh sách bị ảnh hưởng
                if ticker_symbol not in unique_articles[article_id]["affected_tickers"]:
                    unique_articles[article_id]["affected_tickers"].append(
                        ticker_symbol
                    )
            else:
                # Lấy các trường dữ liệu cần thiết theo format của Finnhub
                title = article.get("headline", "")
                summary = article.get("summary", "")

                unix_time = article.get("datetime")
                pub_date = (
                    datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M:%S")
                    if unix_time
                    else ""
                )

                publisher = article.get("source", "")
                link = article.get("url", "")

                # Lưu vào dictionary mới
                unique_articles[article_id] = {
                    "title": title,
                    "summary": summary,
                    "pubDate": pub_date,
                    "publisher": publisher,
                    "link": link,
                    "affected_tickers": [
                        ticker_symbol
                    ],  # Khởi tạo danh sách với Ticker đầu tiên phát hiện được
                }

        # Tịnh tiến ngày bắt đầu của cửa sổ tiếp theo lên 1 ngày sau ngày kết thúc của cửa sổ hiện tại
        current_start_date = current_end_date + timedelta(days=1)

# Lưu lại toàn bộ dữ liệu vào file JSON cục bộ
with open(db_file, "w", encoding="utf-8") as f:
    json.dump(unique_articles, f, ensure_ascii=False, indent=4)

print(f"Hoàn thành. Tổng số bài báo hiện có trong Database: {len(unique_articles)}")
