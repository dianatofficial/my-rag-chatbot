import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from data_loader import load_dataset

# ---------------------------------------------------------------------------
# بارگذاری تنظیمات: ابتدا Streamlit Secrets، سپس .env محلی
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"


def get_config(key: str, default: str | None = None) -> str | None:
"""
مقدار یک کلید تنظیمات را برمی‌گرداند.

ترتیب اولویت:
۱. Streamlit Secrets (محیط Streamlit Cloud)
۲. متغیرهای محیطی / فایل .env (اجرای محلی)
"""
try:
import streamlit as st

if key in st.secrets:
return str(st.secrets[key])
except Exception:
# streamlit نصب نیست، یا secrets.toml وجود ندارد → به سراغ env می‌رویم
pass
return os.getenv(key, default)


MODEL_NAME = get_config("MODEL_NAME", "gemini-3-flash-preview")
EMBED_MODEL = get_config("EMBED_MODEL", "text-embedding-3-small")

PROMPT = ChatPromptTemplate.from_template(
"""تو یک دستیار پژوهشی هستی که فقط بر پایه «متن مرجع» زیر پاسخ می‌دهد.
اگر پاسخ در متن مرجع نبود، صریحاً بگو که اطلاعاتی در دیتاست‌ها پیدا نکردی و حدس نزن.

قواعد نگارش پاسخ:
- پاسخ را به فارسی روان و ساختاریافته بنویس.
- برای داده‌های چندسطری از جدول Markdown استفاده کن.
- برای فرمول‌های ریاضی از $...$ یا $$...$$ استفاده کن.
- اگر داده‌ی عددی قابل مقایسه وجود داشت و نمودار به فهم کمک می‌کرد،
  علاوه بر توضیح متنی یک بلوک با زبان chart اضافه کن، دقیقاً با این ساختار:

```chart
{{"type": "bar", "x": "نام ستون محور افقی", "y": ["ستون عددی"],
  "title": "عنوان نمودار",
  "data": [{{"نام ستون محور افقی": "الف", "ستون عددی": 12}}]}}

- مقدار type یکی از bar یا line یا area یا scatter باشد.
- اگر داده‌ی عددی وجود ندارد، هیچ بلوک chart تولید نکن.

متن مرجع:
{context}

پرسش: {question}

پاسخ:"""
)


def _format_docs(docs) -> str:
    """اسناد بازیابی‌شده را به یک رشته‌ی واحد تبدیل می‌کند."""
    return "\n\n".join(d.page_content for d in docs)


def _load_credentials() -> tuple[str, str]:
    """
    اعتبارنامه‌ها را می‌خواند و اعتبارسنجی می‌کند.

    Raises:
        ValueError: اگر API_KEY یا BASE_URL تنظیم نشده باشد.
    """
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")

    missing = [
        name
        for name, value in (("API_KEY", api_key), ("BASE_URL", base_url))
        if not value
    ]
    if missing:
        raise ValueError(
            f"تنظیمات ناقص است. کلیدهای زیر یافت نشدند: {', '.join(missing)}\n"
            "در اجرای محلی آن‌ها را در فایل .env قرار دهید و در Streamlit Cloud "
            "از بخش Manage app → Settings → Secrets اضافه کنید."
        )

    return api_key, base_url


def get_embeddings() -> OpenAIEmbeddings:
    """
    مدل تبدیل متن به بردار، از طریق endpoint سازگار با OpenAI.

    نکته‌ها:
    - check_embedding_ctx_length=False متن خام را می‌فرستد نه آرایه‌ی توکن.
      بسیاری از سرویس‌های واسطه آرایه‌ی توکن را نمی‌پذیرند و این پرچم
      از آن ناسازگاری جلوگیری می‌کند. چون قطعات ما ۵۰۰ کاراکتری‌اند،
      هیچ خطر سرریز پنجره‌ی متنی وجود ندارد.
    - chunk_size اندازه‌ی هر batch ارسالی است، نه طول متن.
    """
    api_key, base_url = _load_credentials()
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        chunk_size=64,
        timeout=60,
        max_retries=3,
    )


def build_vectorstore(data_dir: str = "data", save: bool = True) -> FAISS:
    """
    ایندکس FAISS را از صفر می‌سازد. این تابع هزینه‌ی API دارد؛
    آن را به‌صورت محلی اجرا کنید، نه در زمان اجرای اپ روی Cloud.
    """
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(rows)
    if not docs:
        raise ValueError("پس از تقسیم‌بندی، هیچ قطعه‌ی متنی تولید نشد.")

    store = FAISS.from_documents(docs, get_embeddings())
    if save:
        store.save_local(str(INDEX_DIR))
    return store


def load_vectorstore() -> FAISS:
    """
    ایندکس پیش‌ساخته را از دیسک می‌خواند.

    allow_dangerous_deserialization فایل را با pickle باز می‌کند و فقط
    برای ایندکسی امن است که خودتان ساخته و کامیت کرده‌اید.
    """
    if not INDEX_DIR.is_dir():
        raise FileNotFoundError(
            f"ایندکس در '{INDEX_DIR}' پیدا نشد.\n"
            "ابتدا به‌صورت محلی «python build_index.py» را اجرا کنید و "
            "پوشه‌ی faiss_index را کامیت و push کنید."
        )
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_rag_chain(data_dir: str = "data", k: int = 5, rebuild: bool = False):
    """
    زنجیره RAG را می‌سازد.

    Args:
        data_dir: مسیر پوشه‌ی دیتاست‌ها (فقط در حالت rebuild استفاده می‌شود).
        k: تعداد قطعات متنی که برای هر پرسش بازیابی می‌شود.
        rebuild: اگر True باشد ایندکس از صفر ساخته می‌شود.

    Returns:
        Runnable که رشته‌ی پرسش می‌گیرد و دیکشنری زیر را برمی‌گرداند:
        {"question": str, "source_documents": list[Document], "answer": str}
    """
    api_key, base_url = _load_credentials()

    vectorstore = build_vectorstore(data_dir) if rebuild else load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=2,
    )

    answer_chain = (
        {
            "context": lambda x: _format_docs(x["source_documents"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # خروجی شامل منابع است تا رابط کاربری بتواند آن‌ها را نمایش دهد
    chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnablePassthrough.assign(answer=answer_chain)

    return chain

## `build_index.py`

python
"""
ساخت ایندکس FAISS. این اسکریپت را روی سیستم خودتان اجرا کنید،
سپس پوشه‌ی faiss_index را کامیت و push کنید.

    python build_index.py
"""

from rag_engine import EMBED_MODEL, INDEX_DIR, build_vectorstore

if __name__ == "__main__":
    print(f"مدل بردارسازی: {EMBED_MODEL}")
    store = build_vectorstore("data", save=True)
    print(f"تعداد بردارها: {store.index.ntotal}")
    print(f"ابعاد بردار: {store.index.d}")
    print(f"ایندکس ذخیره شد در: {INDEX_DIR}")

پیش از اجرا، صحت syntax را بگیرید:

powershell
python -m py_compile rag_engine.py build_index.py
python build_index.py

اگر `app.py` هم تابع `format_docs` را import می‌کند، آن را به `_format_docs` تغییر دهید.

