import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from data_loader import load_dataset

# ---------------------------------------------------------------------------
# بارگذاری تنظیمات: ابتدا .env محلی، سپس Streamlit Secrets در زمان اجرا
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


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

PROMPT = ChatPromptTemplate.from_template(
    """تو یک دستیار پژوهشی هستی که فقط بر پایه «متن مرجع» زیر پاسخ می‌دهد.
اگر پاسخ در متن مرجع نبود، صریحاً بگو که اطلاعاتی در دیتاست‌ها پیدا نکردی و حدس نزن.

متن مرجع:
{context}

پرسش: {question}

پاسخ:"""
)


def _format_docs(docs) -> str:
    """اسناد بازیابی‌شده را به یک رشته‌ی واحد تبدیل می‌کند."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


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


def build_rag_chain(data_dir: str, k: int = 5):
    """
    زنجیره RAG را بر پایه فایل‌های موجود در data_dir می‌سازد.

    Args:
        data_dir: مسیر پوشه‌ی دیتاست‌ها.
        k: تعداد قطعات متنی که برای هر پرسش بازیابی می‌شود.

    Returns:
        یک Runnable که رشته‌ی پرسش می‌گیرد و رشته‌ی پاسخ برمی‌گرداند.
    """
    # ۰. اعتبارسنجی تنظیمات (در زمان فراخوانی، نه در زمان import)
    api_key, base_url = _load_credentials()

    # ۱. بارگذاری تمام فایل‌ها
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")

    # ۲. تقسیم‌بندی متن‌های طولانی به قطعات کوچک
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.create_documents(rows)
    if not docs:
        raise ValueError("پس از تقسیم‌بندی، هیچ قطعه‌ی متنی تولید نشد.")

    # ۳. تبدیل به بردار (مدل چندزبانه، مناسب متن فارسی)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # ۴. اتصال به API اختصاصی
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=2,
    )

    # ۵. زنجیره RAG به سبک LCEL
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
