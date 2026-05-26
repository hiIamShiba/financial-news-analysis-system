import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)


class SentimentModelManager:
    def __init__(self, language="en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Đang khởi động Model trên thiết bị: {self.device} ---")

        if language == "vi":
            # Model tiếng Việt: PhoBERT-VFA
            self.model_name = "kytrungchauwork/phobert-vfa-sentiment"
        else:
            # Model tiếng Anh: FinBERT
            self.model_name = "finance-sentiment/finetuned-FinBERT-sentiment-classification"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def get_sentiment(self, texts):
        """
        Input: list các câu văn bản
        Output: list các dictionary chứa nhãn và điểm số
        """

        if not texts:
            return []

        return self.classifier(texts)


# Test nhanh nếu chạy file này độc lập
if __name__ == "__main__":
    manager = SentimentModelManager(language="en")

    test_text = [
        "Stock market is rising today!",
        "Company declared bankruptcy."
    ]

    print(manager.get_sentiment(test_text))