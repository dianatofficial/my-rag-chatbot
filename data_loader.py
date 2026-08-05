import pandas as pd
from pathlib import Path

# نام دقیق سه فایل شما + سقف تعداد سطرهای خوانده‌شده از هر کدام
# فایل moral_machine_responses بسیار بزرگ است، پس فقط 5000 سطر اول را می‌خوانیم
# دو فایل دیگر کوچک هستند و کامل خوانده می‌شوند (None یعنی بدون محدودیت)
DATASET_FILES = {
    "country_preferences.csv": None,
    "demographic_preferences.csv": None,
    "moral_machine_responses.csv": 5000,
}


def load_dataset(data_dir: str) -> list[str]:
    """هر سه فایل CSV را می‌خواند و هر سطر را به یک متن قابل فهم تبدیل می‌کند."""
    data_path = Path(data_dir)
    texts = []

    for filename, row_limit in DATASET_FILES.items():
        file_path = data_path / filename

        # اگر فایلی وجود نداشت، برنامه کرش نمی‌کند؛ فقط هشدار می‌دهد و رد می‌شود
        if not file_path.exists():
            print(f"هشدار: فایل {filename} پیدا نشد و نادیده گرفته شد.")
            continue

        df = pd.read_csv(file_path, nrows=row_limit)

        # حذف سطرهایی که کاملاً خالی هستند
        df = df.dropna(how="all")

        # نام فایل بدون پسوند — مثلا: country_preferences
        source_name = file_path.stem

        # هر سطر جدول تبدیل می‌شود به متنی مثل:
        # [country_preferences] Country: Iran | Preference: 0.42 | ...
        texts += df.apply(
            lambda row: f"[{source_name}] "
            + " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)),
            axis=1,
        ).tolist()

        print(f"فایل {filename} خوانده شد: {len(df)} سطر")

    if not texts:
        raise FileNotFoundError(
            f"هیچ فایل CSV در پوشه '{data_dir}' پیدا نشد. "
            "مطمئن شوید سه فایل داخل پوشه data قرار دارند."
        )

    return texts
