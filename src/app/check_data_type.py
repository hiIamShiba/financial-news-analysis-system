import chromadb
import pandas as pd


def inspect_chroma_db(path="./chroma_db"):

    # 1. Kết nối tới database
    client = chromadb.PersistentClient(path=path)

    # 2. Lấy danh sách tất cả collections
    collections = client.list_collections()

    if not collections:
        print("Không tìm thấy collection nào trong database này.")
        return

    print(f"--- Tìm thấy {len(collections)} collection(s) ---")

    for col_info in collections:

        collection = client.get_collection(
            name=col_info.name
        )

        count = collection.count()

        print(f"\n[Collection Name]: {col_info.name}")
        print(f" - Số lượng bản ghi: {count}")

        if count > 0:

            # 3. Lấy thử 3 bản ghi đầu tiên để xem cấu trúc

            sample = collection.get(limit=3)

            print(
                " - Metadata keys (Dùng để lọc):",
                sample['metadatas'][0].keys()
                if sample['metadatas']
                else "Không có metadata"
            )

            print(" - Nội dung mẫu (Document):")

            for i in range(len(sample['documents'])):

                doc = sample['documents'][i]

                meta = (
                    sample['metadatas'][i]
                    if sample['metadatas']
                    else {}
                )

                # In rút gọn nội dung

                print(
                    f" {i+1}. "
                    f"[{meta.get('ticker', 'N/A')}] "
                    f"{doc[:100]}..."
                )

        else:
            print(" - Collection này đang trống.")


if __name__ == "__main__":

    # Đảm bảo đường dẫn này đúng với folder chroma_db

    inspect_chroma_db("./chroma_db")