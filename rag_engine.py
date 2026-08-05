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

# مسیر مطلق به فایل .env (فرقی نمی‌کند از کجا اجرا کنی)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# بررسی اینکه متغیرها خوانده شدند یا نه
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise ValueError("خطا: API_KEY یا BASE_URL در فایل .env تعریف نشده است")

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


def build_rag_chain(data_dir: str, k: int = 5):
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

    # ۳. تبدیل به بردار (مدل چندزبانه، مناسب متن فارسی)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # ۴. اتصال به API اختصاصی
    llm = ChatOpenAI(
        model="gemini-3-flash-preview",
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )

    # ۵. زنجیره RAG به سبک LCEL
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
