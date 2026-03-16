import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuMind Pro",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocuMind Pro")
st.caption("Your Private Knowledge Intelligence Engine")

# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = None


# -----------------------------
# BACKEND HEALTH CHECK
# -----------------------------

if st.session_state.backend_ok is None:
    try:
        r = requests.get(f"{API}/health", timeout=3)

        if r.status_code == 200:
            st.session_state.backend_ok = True
        else:
            st.session_state.backend_ok = False

    except Exception:
        st.session_state.backend_ok = False


# -----------------------------
# SIDEBAR - PDF UPLOAD
# -----------------------------

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Upload & Process"):

        if not st.session_state.backend_ok:
            st.error("Backend not running")
        else:

            for uf in uploaded_files:

                files = {
                    "file": (uf.name, uf.getvalue(), "application/pdf")
                }

                try:
                    r = requests.post(
                        f"{API}/upload",
                        files=files,
                        timeout=60
                    )

                    if r.status_code != 200:
                        st.error(f"Upload failed: {r.text}")

                except Exception as e:
                    st.error(f"Upload failed: {e}")

            st.success("Upload complete")


# -----------------------------
# CHAT HISTORY
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))


# -----------------------------
# USER INPUT
# -----------------------------

prompt = st.chat_input("Ask something about your documents...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if not st.session_state.backend_ok:

            answer = "Backend not running."
            sources = []

            st.error(answer)

        else:

            with st.spinner("Thinking..."):

                try:

                    res = requests.get(
                        f"{API}/ask",
                        params={"q": prompt},
                        timeout=60
                    )

                    if res.status_code != 200:
                        st.error(f"Backend error: {res.status_code}")
                        st.write(res.text)

                        answer = "Backend error"
                        sources = []

                    else:

                        data = res.json()

                        answer = data.get("answer", "No response")
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            st.caption("Sources: " + ", ".join(sources))

                except Exception as e:

                    answer = "Backend request failed"
                    sources = []

                    st.error(e)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })