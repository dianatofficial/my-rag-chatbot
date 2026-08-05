import streamlit as st
from rag_engine import build_rag_chain

st.set_page_config(page_title="دستیار دیتاست‌های اخلاق", page_icon="🤖")
st.title("🤖 دستیار هوشمند دیتاست‌های اخلاق")


# بارگذاری موتور (فقط یک بار انجام می‌شود تا سرعت حفظ شود)
@st.cache_resource(show_spinner="در حال ساخت ایندکس دیتاست‌ها...")
def get_chain():
    return build_rag_chain("data")


try:
    chain = get_chain()
except Exception as e:
    st.error(f"خطا در راه‌اندازی موتور: {e}")
    st.stop()


# مدیریت پیام‌های چت
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# دریافت سوال
if prompt := st.chat_input("سوال خود را درباره دیتاست‌ها بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("در حال جستجو در فایل‌ها..."):
            try:
                response = chain.invoke(prompt)
            except Exception as e:
                response = f"خطا در تولید پاسخ: {e}"
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
