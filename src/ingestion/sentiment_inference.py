import json
import torch
import os
from transformers import pipeline

# Xác định đường dẫn thư mục gốc (root) của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn file
INPUT_JSON = os.path.join(BASE_DIR, "data", "raw", "bert_news_db.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "processed", "news_with_sentiment.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "finbert_fine_tuned")


def main():
    # Load dữ liệu
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Khởi tạo pipeline trên GPU (cuda:0)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, device=device
    )

    # Chuẩn bị dữ liệu cho batch inference
    news_ids = list(data.keys())
    texts_to_infer = []

    for news_id in news_ids:
        item = data[news_id]
        # Kết hợp title và summary để có ngữ cảnh tốt nhất
        combined_text = f"{item['title']}. {item['summary']}"
        texts_to_infer.append(combined_text)

    # Chạy inference theo batch (RTX 3060 12GB có thể chạy batch 16-32 tùy độ dài chuỗi)
    print("Bắt đầu chạy FinBERT inference...")
    results = sentiment_pipeline(
        texts_to_infer, batch_size=16, truncation=True, max_length=512
    )

    # Map kết quả trở lại JSON
    for news_id, sentiment_result in zip(news_ids, results):
        # Đảm bảo nhãn ở định dạng text chuẩn (Positive, Negative, Neutral)
        label = sentiment_result["label"].capitalize()
        data[news_id]["sentiment"] = label
        data[news_id]["sentiment_score"] = sentiment_result["score"]

    # Lưu file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Đã lưu dữ liệu kèm sentiment vào {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
