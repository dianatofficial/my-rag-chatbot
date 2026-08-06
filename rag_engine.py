import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover - optional dependency
    HuggingFaceEmbeddings = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    OpenAIEmbeddings = None

from data_loader import load_dataset

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"


def get_config(key: str, default: str | None = None) -> str | None:
    """مقدار یک کلید تنظیمات را از Streamlit Secrets یا .env برمی‌گرداند."""
    fallbacks = [key]
    if key == "API_KEY":
        fallbacks.append("OPENAI_API_KEY")
    elif key == "BASE_URL":
        fallbacks.append("OPENAI_BASE_URL")

    try:
        import streamlit as st

        for name in fallbacks:
            if name in st.secrets:
                return str(st.secrets[name])
    except Exception:
        pass

    for name in fallbacks:
        value = os.getenv(name)
        if value:
            return value

    return default


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


def _load_credentials() -> tuple[str | None, str | None]:
    """اعتبارنامه‌ها را می‌خواند و در صورت نبود، None برمی‌گرداند."""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    return api_key, base_url


def get_embeddings():
    """مدل تبدیل متن به بردار؛ در حالت پیش‌فرض از HuggingFace محلی استفاده می‌شود."""
    api_key, base_url = _load_credentials()
    if os.getenv("USE_OPENAI_EMBEDDINGS") == "1" and api_key and base_url and OpenAIEmbeddings is not None:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        return OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_key=api_key,
            openai_api_base=base_url,
            check_embedding_ctx_length=False,
            chunk_size=64,
            timeout=60,
            max_retries=3,
        )

    return get_local_embeddings()


def get_local_embeddings():
    """مدل بردار محلی برای حالت fallback."""
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            "بسته‌ی langchain-huggingface در این محیط نصب نیست. "
            "برای اجرای محلی، بسته را با pip install -r requirements.txt نصب کنید."
        )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(data_dir: str = "data", save: bool = True) -> FAISS:
    """ایندکس FAISS را از صفر می‌سازد. هزینه‌ی API دارد؛ محلی اجرا شود."""
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(rows)
    if not docs:
        raise ValueError("پس از تقسیم‌بندی، هیچ قطعه‌ی متنی تولید نشد.")

    embeddings = get_embeddings()
    try:
        store = FAISS.from_documents(docs, embeddings)
    except Exception:
        fallback_embeddings = get_local_embeddings()
        store = FAISS.from_documents(docs, fallback_embeddings)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(INDEX_DIR))
    return store


def load_vectorstore(data_dir: str = "data", force_build: bool = False) -> FAISS:
    """ایندکس پیش‌ساخته را از دیسک می‌خواند و در صورت نبود، آن را می‌سازد."""
    if force_build or not INDEX_DIR.exists():
        return build_vectorstore(data_dir, save=True)

    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return build_vectorstore(data_dir, save=True)


def get_llm(api_key: str | None, base_url: str | None):
    """LLM را در صورت امکان از OpenAI می‌سازد و در غیر این صورت از HuggingFace."""
    if ChatOpenAI is not None and api_key and base_url:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            api_key=api_key,
            base_url=base_url,
            timeout=60,
            max_retries=2,
        )
        try:
            llm.invoke("test")
            return llm
        except Exception:
            pass

    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "برای اجرای LLM محلی، بسته‌های transformers و langchain-community لازم‌اند."
        ) from exc

    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,
    )
    return HuggingFacePipeline(pipeline=generator)


def build_rag_chain(data_dir: str = "data", k: int = 5, rebuild: bool = False):
    """زنجیره RAG را می‌سازد و خروجی همراه با منابع برمی‌گرداند."""
    api_key, base_url = _load_credentials()

    vectorstore = (
        build_vectorstore(data_dir)
        if rebuild
        else load_vectorstore(data_dir=data_dir, force_build=not INDEX_DIR.exists())
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = get_llm(api_key, base_url)

    answer_chain = (
        {
            "context": lambda x: _format_docs(x["source_documents"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnablePassthrough.assign(answer=answer_chain)

    return chain
