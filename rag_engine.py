import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover - optional dependency
    HuggingFaceEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    OpenAIEmbeddings = None

from data_loader import load_dataset

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class FallbackLLM:
    """LLM جایگزین برای مواقعی که مدل اصلی در دسترس نیست."""

    def invoke(self, value):
        prompt = str(value)
        if not prompt:
            return "پاسخ جایگزین: اطلاعات موجود در دیتاست‌ها را از روی متن مرجع استخراج کنید."

        return (
            "پاسخ جایگزین: متن مرجع برای این پرسش در دسترس نیست یا مدل اصلی پاسخ نمی‌دهد. "
            "به‌جای حدس زدن، از داده‌های موجود در دیتاست‌ها برای تولید پاسخ استفاده کنید."
        )

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


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


MODEL_NAME = get_config("MODEL_NAME", "gpt-4o-mini")
EMBED_MODEL = get_config("EMBED_MODEL", "text-embedding-3-large")


def _use_openai_embeddings() -> bool:
    value = str(get_config("USE_OPENAI_EMBEDDINGS", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}

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


def _build_documents(rows: list[str]):
    """برای داده‌های سطری CSV فقط متن‌های بلند را chunk می‌کند تا تعداد embedding بی‌جهت زیاد نشود."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    documents = []

    for row in rows:
        source = "unknown"
        if row.startswith("[") and "]" in row:
            source = row[1 : row.index("]")]

        if len(row) <= 1200:
            documents.append(Document(page_content=row, metadata={"source": source}))
            continue

        documents.extend(splitter.create_documents([row], metadatas=[{"source": source}]))

    return documents


def _load_credentials() -> tuple[str | None, str | None]:
    """اعتبارنامه‌ها را می‌خواند و در صورت نبود، None برمی‌گرداند."""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")
    return api_key, base_url


def get_embeddings():
    """مدل تبدیل متن به بردار؛ در حالت پیش‌فرض از OpenAI-compatible API استفاده می‌شود."""
    api_key, base_url = _load_credentials()
    if _use_openai_embeddings() and api_key and base_url and OpenAIEmbeddings is not None:
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
    if HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    if SentenceTransformer is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        class _LocalEmbeddings:
            dimension = 384

            def __init__(self, model):
                self.model = model

            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()

            def embed_query(self, text):
                return self.model.encode([text], convert_to_numpy=True)[0].tolist()

            def __call__(self, text):
                return self.embed_query(text)

        return _LocalEmbeddings(model)

    class _FallbackEmbeddings:
        dimension = 8

        def __init__(self):
            self._dimension = 8

        def _simple_vector(self, text: str) -> list[float]:
            text = (text or "").lower()
            vector = [0.0] * self._dimension
            for index, ch in enumerate(text):
                vector[index % self._dimension] += (ord(ch) % 11) / 10.0
            return vector

        def embed_documents(self, texts):
            return [self._simple_vector(text) for text in texts]

        def embed_query(self, text):
            return self._simple_vector(text)

        def __call__(self, text):
            return self.embed_query(text)

    return _FallbackEmbeddings()


def _embedding_dimension(embeddings) -> int | None:
    """ابعاد embedding را بدون درخواست شبکه‌ای برمی‌گرداند."""
    dimension = getattr(embeddings, "dimension", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension

    dimensions = getattr(embeddings, "dimensions", None)
    if isinstance(dimensions, int) and dimensions > 0:
        return dimensions

    model_name = getattr(embeddings, "model", None)
    if isinstance(model_name, str):
        return OPENAI_EMBEDDING_DIMENSIONS.get(model_name)

    return None


def _read_index_metadata() -> dict:
    if not INDEX_METADATA_PATH.exists():
        return {}

    try:
        return json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_index_metadata(embeddings, store: FAISS) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_backend": type(embeddings).__name__,
        "embedding_model": getattr(embeddings, "model", None),
        "dimension": getattr(store.index, "d", None),
        "vector_count": getattr(store.index, "ntotal", None),
    }
    INDEX_METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _select_embeddings_for_index(index_dim: int | None):
    """embedding سازگار با بعد ایندکس را انتخاب می‌کند."""
    metadata = _read_index_metadata()
    metadata_dim = metadata.get("dimension")
    if isinstance(metadata_dim, int) and metadata_dim > 0:
        index_dim = metadata_dim

    candidates = []

    try:
        candidates.append(get_embeddings())
    except Exception:
        pass

    try:
        candidates.append(get_local_embeddings())
    except Exception:
        pass

    for embeddings in candidates:
        dim = _embedding_dimension(embeddings)
        if index_dim is not None and dim == index_dim:
            return embeddings

    if candidates:
        return candidates[0]

    return get_local_embeddings()


def _preferred_embeddings():
    """embedding ترجیحی پروژه را بر اساس تنظیمات جاری برمی‌گرداند."""
    return get_embeddings()


def _should_rebuild_index(index_dim: int | None, preferred_embeddings) -> bool:
    """اگر ایندکس فعلی با embedding ترجیحی ناسازگار باشد، باید بازسازی شود."""
    preferred_dim = _embedding_dimension(preferred_embeddings)
    if index_dim is None or preferred_dim is None:
        return False

    if not _use_openai_embeddings():
        return False

    api_key, base_url = _load_credentials()
    if not (api_key and base_url and OpenAIEmbeddings is not None):
        return False

    return index_dim != preferred_dim


def build_vectorstore(data_dir: str = "data", save: bool = True) -> FAISS:
    """ایندکس FAISS را از صفر می‌سازد. هزینه‌ی API دارد؛ محلی اجرا شود."""
    rows = load_dataset(data_dir)
    if not rows:
        raise ValueError(f"هیچ داده‌ای در مسیر '{data_dir}' پیدا نشد.")

    docs = _build_documents(rows)
    if not docs:
        raise ValueError("پس از تقسیم‌بندی، هیچ قطعه‌ی متنی تولید نشد.")

    embeddings = get_embeddings()
    try:
        store = FAISS.from_documents(docs, embeddings)
    except Exception:
        fallback_embeddings = get_local_embeddings()
        embeddings = fallback_embeddings
        store = FAISS.from_documents(docs, fallback_embeddings)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(INDEX_DIR))
        _write_index_metadata(embeddings, store)
    return store


def load_vectorstore(data_dir: str = "data", force_build: bool = False) -> FAISS:
    """ایندکس پیش‌ساخته را از دیسک می‌خواند و در صورت نبود، آن را می‌سازد."""
    if force_build or not INDEX_DIR.exists():
        return build_vectorstore(data_dir, save=True)

    index_dim = None
    try:
        import faiss

        index_path = INDEX_DIR / "index.faiss"
        if index_path.exists():
            index_dim = faiss.read_index(str(index_path)).d
    except Exception:
        index_dim = None

    try:
        preferred_embeddings = _preferred_embeddings()
    except Exception:
        preferred_embeddings = None

    if preferred_embeddings is not None and _should_rebuild_index(index_dim, preferred_embeddings):
        return build_vectorstore(data_dir, save=True)

    embeddings = _select_embeddings_for_index(index_dim)
    try:
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # اگر بعد embedding با بعد ایندکس نخواند، به‌جای کرش‌کردن ایندکس بازسازی می‌شود.
        dim = _embedding_dimension(embeddings)
        if dim is not None and getattr(store.index, "d", None) not in (None, dim):
            return build_vectorstore(data_dir, save=True)

        return store
    except Exception:
        return build_vectorstore(data_dir, save=True)


def get_llm(api_key: str | None, base_url: str | None):
    """LLM را در صورت امکان از OpenAI می‌سازد و در غیر این صورت fallback سبک برمی‌گرداند."""
    if ChatOpenAI is not None and api_key and base_url:
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            api_key=api_key,
            base_url=base_url,
            timeout=60,
            max_retries=2,
        )

    return FallbackLLM()


def build_rag_chain(
    data_dir: str = "data",
    k: int = 5,
    rebuild: bool = False,
    vectorstore: FAISS | None = None,
):
    """زنجیره RAG را می‌سازد و خروجی همراه با منابع برمی‌گرداند."""
    api_key, base_url = _load_credentials()

    vectorstore = vectorstore or (
        build_vectorstore(data_dir)
        if rebuild
        else load_vectorstore(data_dir=data_dir, force_build=not INDEX_DIR.exists())
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    try:
        llm = get_llm(api_key, base_url)
    except Exception:
        llm = FallbackLLM()

    def build_answer(inputs):
        prompt_value = PROMPT.invoke(
            {
                "context": _format_docs(inputs["source_documents"]),
                "question": inputs["question"],
            }
        )
        try:
            raw = llm.invoke(prompt_value)
            return StrOutputParser().invoke(raw)
        except Exception:
            return (
                "پاسخ جایگزین: متن مرجع برای این پرسش در دسترس نیست یا مدل اصلی پاسخ نمی‌دهد. "
                "برای ادامه، از داده‌های موجود در دیتاست‌ها استفاده کنید."
            )

    def build_result(inputs):
        answer = build_answer(inputs)
        return {
            "question": inputs["question"],
            "source_documents": inputs["source_documents"],
            "answer": answer,
        }

    chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ) | RunnableLambda(build_result)

    return chain
