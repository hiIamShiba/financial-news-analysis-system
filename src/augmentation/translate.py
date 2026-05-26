import os
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ================= INIT =================
load_dotenv()
client = genai.Client()

INPUT_FILE = "source.json"
OUTPUT_FILE = "translated_source.json"
BATCH_SIZE = 30
SLEEP_TIME = 1

# ================= PROMPT =================
PROMPT = """
You are a professional financial translator with strong expertise in finance, economics, and business terminology.

Your task is to translate a JSON dataset from English to Vietnamese.

Input format:
[
  {
    "Sentence": "...",
    "Sentiment": "positive | negative | neutral"
  }
]

Requirements:
- Translate "Sentence" into fluent Vietnamese with correct financial terminology.
- Preserve meaning, tone, nuance.
- Keep "Sentiment" unchanged.
- Return valid JSON ONLY.
- No explanation, no extra text.
"""


# ================= UTILS =================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunk_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


# ================= GEMINI CALL =================
def translate_batch(batch):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    f"Translate the following JSON:\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
                ],
                config=types.GenerateContentConfig(
                    system_instruction=PROMPT,
                    temperature=0.0,
                    top_p=0.95,
                    max_output_tokens=65536,
                    response_mime_type="application/json",
                ),
            )

            return json.loads(response.text)

        except json.JSONDecodeError:
            print(f"-> JSON lỗi (lần {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                return []

        except Exception as e:
            error_msg = str(e).lower()

            if "429" in error_msg or "quota" in error_msg:
                print("-> Quota exceeded")
                raise RuntimeError("QUOTA_EXHAUSTED") from e

            print(f"-> Lỗi API (lần {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return []

        time.sleep(2)


# ================= MAIN =================
def main():
    if not os.path.exists(INPUT_FILE):
        print("Không tìm thấy input.json")
        return

    data = load_json(INPUT_FILE)
    results = []
    quota_exhausted = False
    processed_count = 0

    total_samples = len(data)

    for i, batch in enumerate(chunk_data(data, BATCH_SIZE)):
        if quota_exhausted:
            break

        print(f"\nĐang xử lý batch {i+1} ({len(batch)} samples)")

        try:
            translated = translate_batch(batch)

            if isinstance(translated, list):
                results.extend(translated)

                processed_count += len(translated)

                print(f"-> Tiến độ: {processed_count}/{total_samples} samples")

                # Check mismatch
                if len(translated) != len(batch):
                    print("-> WARNING: mismatch số lượng output")

            else:
                print("-> Output không phải list")

        except RuntimeError as e:
            if str(e) == "QUOTA_EXHAUSTED":
                quota_exhausted = True

                print(f"\n-> HẾT QUOTA tại batch {i+1}")
                print(f"-> Đã xử lý: {processed_count}/{total_samples} samples")

                # Save partial
                if results:
                    partial_path = OUTPUT_FILE.replace(".json", "_partial.json")
                    save_json(results, partial_path)
                    print(f"-> Đã lưu tạm: {partial_path}")

        # Save checkpoint mỗi 5 batch
        if (i + 1) % 5 == 0:
            checkpoint_path = OUTPUT_FILE.replace(".json", "_checkpoint.json")
            save_json(results, checkpoint_path)
            print("-> Đã lưu checkpoint")

        time.sleep(SLEEP_TIME)

    # Save final nếu không bị quota
    if not quota_exhausted:
        if results:
            save_json(results, OUTPUT_FILE)
            print(f"\n-> Đã lưu hoàn chỉnh: {OUTPUT_FILE}")
        else:
            print("-> Không có dữ liệu")

    if quota_exhausted:
        print("-> Kết thúc sớm do hết quota")
        sys.exit(1)


if __name__ == "__main__":
    main()
