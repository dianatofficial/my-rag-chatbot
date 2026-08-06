import json
import re
import traceback

import pandas as pd
import streamlit as st

from rag_engine import build_rag_chain

# ---------------------------------------------------------------- تنظیمات صفحه
st.set_page_config(
    page_title="دستیار دیتاست‌های اخلاق",
    page_icon="🤖",
    layout="centered",
)

# فونت وزیرمتن از CDN؛ در نبود دسترسی به اینترنت به Tahoma برمی‌گردد
st.markdown(
    """
<!-- Streamlit UI styling -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css">
<style>
  :root { --app-font: "Vazirmatn", Tahoma, sans-serif; }

  html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
      font-family: var(--app-font) !important;
  }

  [data-testid="stAppViewContainer"] [data-testid="stChatMessage"],
  [data-testid="stAppViewContainer"] [data-testid="stChatMessage"] * {
      direction: rtl !important;
      text-align: right !important;
  }

  [data-testid="stChatInput"] textarea {
      direction: rtl !important;
      text-align: right !important;
      font-family: var(--app-font) !important;
  }

  code, pre, [data-testid="stCodeBlock"],
  [data-testid="stDataFrame"], [data-testid="stTable"],
  .stPlotlyChart, .stVegaLiteChart, canvas {
      direction: ltr !important;
      text-align: left !important;
  }

  [data-testid="stChatMessage"] p { line-height: 1.9 !important; }

  #MainMenu, footer { display: none !important; }
</style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Engine — تحلیل داده")
st.caption("پاسخ‌ها فقط بر پایه‌ی محتوای دیتاست‌های موجود تولید می‌شوند.")

if "rebuild_index" not in st.session_state:
    st.session_state.rebuild_index = 0


# ------------------------------------------------------------- بارگذاری زنجیره
@st.cache_resource(show_spinner="در حال بارگذاری ایندکس دیتاست‌ها...")
def get_chain(k: int, rebuild: int = 0):
    """زنجیره‌ی RAG. فقط یک‌بار در طول عمر کانتینر ساخته می‌شود."""
    return build_rag_chain("data", k=k, rebuild=bool(rebuild))


# ------------------------------------------------------------------- نمایش غنی
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _render_chart(payload: str) -> None:
    """یک بلوک chart را به نمودار Streamlit تبدیل می‌کند."""
    spec = json.loads(payload)
    df = pd.DataFrame(spec["data"])
    if df.empty:
        st.info("داده‌ای برای رسم نمودار وجود ندارد.")
        return

    if spec.get("title"):
        st.markdown(f"**{spec['title']}**")

    kind = spec.get("type", "bar")
    x = spec.get("x") or df.columns[0]
    y = spec.get("y") or [c for c in df.columns if c != x]

    renderers = {
        "bar": st.bar_chart,
        "line": st.line_chart,
        "area": st.area_chart,
        "scatter": st.scatter_chart,
    }
    renderers.get(kind, st.bar_chart)(df, x=x, y=y)

    with st.expander("داده‌ی نمودار"):
        st.dataframe(df, use_container_width=True)


def render_rich(text: str) -> None:
    """
    متن پاسخ را رندر می‌کند و بلوک‌های کد را جداگانه پردازش می‌کند:
      - chart → نمودار
      - json حاوی آرایه → جدول
      - سایر زبان‌ها → کد با هایلایت
    جدول Markdown و فرمول‌های LaTeX توسط خود st.markdown پشتیبانی می‌شوند.
    """
    cursor = 0
    for match in FENCE.finditer(text):
        prefix = text[cursor : match.start()].strip()
        if prefix:
            st.markdown(prefix)

        lang = (match.group(1) or "").lower()
        body = match.group(2).strip()

        if lang == "chart":
            try:
                _render_chart(body)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"بلوک نمودار قابل رسم نبود: {exc}")
                st.code(body, language="json")
        elif lang == "json":
            try:
                data = json.loads(body)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                else:
                    st.json(data)
            except Exception:  # noqa: BLE001
                st.code(body, language="json")
        else:
            st.code(body, language=lang or None)

        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        st.markdown(tail)


# --------------------------------------------------------------------- نوار کناری
with st.sidebar:
    st.subheader("تنظیمات")
    top_k = st.slider(
        "تعداد قطعات بازیابی‌شده",
        min_value=2,
        max_value=12,
        value=5,
        help="مقدار بیشتر پوشش را بالا می‌برد ولی هزینه و زمان پاسخ را افزایش می‌دهد.",
    )
    show_sources = st.toggle("نمایش منابع پاسخ", value=True)

    if st.button("بازسازی ایندکس", use_container_width=True):
        st.session_state.rebuild_index += 1
        st.rerun()

    if st.button("پاک‌کردن تاریخچه", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("ایندکس در حافظه کش می‌شود؛ پاک‌کردن تاریخچه آن را بازنمی‌سازد.")


try:
    chain = get_chain(top_k, rebuild=st.session_state.rebuild_index)
except Exception as exc:  # noqa: BLE001
    st.error(
        "موتور آماده نیست. در حال حاضر ایندکس بازسازی یا بارگذاری نمی‌شود. "
        "چند لحظه صبر کنید یا دوباره دکمه «بازسازی ایندکس» را بزنید."
    )
    with st.expander("جزئیات فنی"):
        st.code(f"{exc}\n\n{traceback.format_exc()}", language="text")
    st.stop()


# ------------------------------------------------------------------ تاریخچه‌ی چت
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_rich(msg["content"])
        else:
            st.markdown(msg["content"])


# -------------------------------------------------------------------- ورودی کاربر
if prompt := st.chat_input("سوال خود را درباره‌ی دیتاست‌ها بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = None
        sources = []

        with st.spinner("در حال جست‌وجو در دیتاست‌ها..."):
            try:
                result = chain.invoke(prompt)
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as exc:  # noqa: BLE001
                st.error(f"خطا در تولید پاسخ: {exc}")
                with st.expander("جزئیات فنی"):
                    st.code(traceback.format_exc(), language="text")

        if answer is not None:
            render_rich(answer)

            if show_sources and sources:
                with st.expander(f"منابع ({len(sources)} قطعه)"):
                    for i, doc in enumerate(sources, start=1):
                        src = doc.metadata.get("source", "نامشخص")
                        st.markdown(f"**{i}. {src}**")
                        st.caption(doc.page_content[:400] + "…")

            # فقط پاسخ‌های معتبر وارد تاریخچه می‌شوند
            st.session_state.messages.append({"role": "assistant", "content": answer})
