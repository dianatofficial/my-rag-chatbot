from rag_engine import EMBED_MODEL, INDEX_DIR, build_vectorstore

if __name__ == "__main__":
    print(f"مدل بردارسازی: {EMBED_MODEL}")
    store = build_vectorstore("data", save=True)
    print(f"تعداد بردارها: {store.index.ntotal}")
    print(f"ابعاد بردار: {store.index.d}")
    print(f"ایندکس ذخیره شد در: {INDEX_DIR}")
