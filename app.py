import traceback

import streamlit as st

from rag_engine import build_rag_chain

# ---------------------------------------------------------------- تنظیمات صفحه
st.set_page_config(
    page_title="دستیار دیتاست‌های اخلاق",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp, .stChatMessage, .stMarkdown, .stTextInput { direction: rtl; text-align: right; }
        .stChatInput textarea { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 دستیار هوشمند دیتاست‌های اخلاق")


# ------------------------------------------------------------- بارگذاری زنجیره
@st.cache_resource(show_spinner="در حال ساخت ایندکس دیتاست‌ها...")
def get_chain():
    """ساخت زنجیرهٔ RAG. فقط یک‌بار در طول عمر نشست اجرا می‌شود."""
    return build_rag_chain("data")


try:
    chain = get_chain()
except Exception as exc:  # noqa: BLE001 - خطای راه‌اندازی باید به کاربر نمایش داده شود
    st.error(f"خطا در راه‌اندازی موتور: {exc}")
    with st.expander("جزئیات فنی"):
        st.code(traceback.format_exc(), language="text")
    st.stop()


# ------------------------------------------------------------------- ابزارهای کمکی
def extract_answer(raw) -> str:
    """نرمال‌سازی خروجی زنجیره به رشتهٔ قابل نمایش."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("answer", "result", "output_text", "output"):
            if key in raw:
                return str(raw[key])
    return str(raw)


def extract_sources(raw) -> list:
    """استخراج اسناد بازیابی‌شده در صورت وجود."""
    if isinstance(raw, dict):
        return raw.get("source_documents", []) or []
    return []


# --------------------------------------------------------------------- نوار کناری
with st.sidebar:
    st.subheader("تنظیمات")
    if st.button("پاک‌کردن تاریخچهٔ گفت‌وگو", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("ایندکس در حافظهٔ سرور کش شده است؛ پاک‌کردن تاریخچه آن را بازنمی‌سازد.")


# ------------------------------------------------------------------ تاریخچهٔ چت
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------------------------------------------------- ورودی کاربر
if prompt := st.chat_input("سوال خود را دربارهٔ دیتاست‌ها بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        answer = None
        sources = []
        with st.spinner("در حال جست‌وجو در فایل‌ها..."):
            try:
                raw = chain.invoke(prompt)
                answer = extract_answer(raw)
                sources = extract_sources(raw)
            except Exception as exc:  # noqa: BLE001
                st.error(f"خطا در تولید پاسخ: {exc}")
                with st.expander("جزئیات فنی"):
                    st.code(traceback.format_exc(), language="text")

        if answer is not None:
            st.write(answer)

            if sources:
                with st.expander(f"منابع ({len(sources)} قطعه)"):
                    for i, doc in enumerate(sources, start=1):
                        src = doc.metadata.get("source", "نامشخص")
                        st.markdown(f"**{i}. {src}**")
                        st.caption(doc.page_content[:400] + "...")

            # فقط پاسخ‌های معتبر وارد تاریخچه می‌شوند
            st.session_state.messages.append({"role": "assistant", "content": answer})
