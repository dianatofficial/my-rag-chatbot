import sys

from rag_engine import build_rag_chain, load_vectorstore

QUESTIONS = [
    "خلاصه ای از داده های موجود بده",
    "بیشترین مقدار عددی در دیتاست مربوط به چیست؟",
]


def check_index() -> bool:
    try:
        store = load_vectorstore()
    except Exception as exc:
        print("[FAIL] load_vectorstore:", exc)
        return False
    print("[OK] index loaded, vectors =", store.index.ntotal)
    return True


def main() -> int:
    if not check_index():
        return 1

    try:
        chain = build_rag_chain(k=5)
    except Exception as exc:
        print("[FAIL] build_rag_chain:", exc)
        return 1
    print("[OK] chain built")

    for q in QUESTIONS:
        print("\n" + "=" * 60)
        print("Q:", q)
        try:
            result = chain.invoke(q)
        except Exception as exc:
            print("[FAIL] invoke:", type(exc).__name__, exc)
            return 1

        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        docs = result["source_documents"]
        print("sources =", len(docs))
        for i, d in enumerate(docs, 1):
            preview = d.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {preview}")

    print("\n[DONE] all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
