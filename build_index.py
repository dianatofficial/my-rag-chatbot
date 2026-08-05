"""
ساخت ایندکس FAISS به صورت لوکال.
خروجی در پوشه faiss_index/ ذخیره می‌شود و باید در گیت کامیت شود
تا روی Streamlit Cloud نیازی به ساخت مجدد (و مصرف RAM) نباشد.

اجرا:  python build_index.py
"""

import sys
import time
import traceback
from pathlib import Path

from rag_engine import build_vectorstore

DATA_DIR = "data"
INDEX_DIR = Path("faiss_index")


def main() -> int:
    if not Path(DATA_DIR).is_dir():
        print(f"خطا: پوشه '{DATA_DIR}' پیدا نشد. از ریشه پروژه اجرا کنید.")
        return 1

    print("شروع ساخت ایندکس...")
    start = time.perf_counter()

    try:
        vs = build_vectorstore(data_dir=DATA_DIR, save=True)
    except Exception:
        print("ساخت ایندکس شکست خورد:\n")
        traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - start
    count = getattr(vs.index, "ntotal", "نامشخص")

    print(f"\nایندکس ساخته شد در {elapsed:.1f} ثانیه")
    print(f"تعداد بردارها: {count}")
    print(f"مسیر: {INDEX_DIR.resolve()}")

    if INDEX_DIR.is_dir():
        for f in sorted(INDEX_DIR.iterdir()):
            print(f"  - {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    else:
        print("هشدار: پوشه faiss_index ساخته نشد. مقدار save در rag_engine را بررسی کنید.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
