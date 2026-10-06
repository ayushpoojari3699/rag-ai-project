"""DocuMind Streamlit chat UI (talks to the FastAPI backend)."""

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")
st.title("🧠 DocuMind")
st.caption("Ask questions about your PDFs. Answers cite the pages they came from.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def backend_ok() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def show_sources(sources, timings=None):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(
                    f"**[{s['id']}] {s['file']} – page {s['page']}** "
                    f"· similarity {s['score']:.2f}"
                )
                st.caption(s["snippet"] + "…")
    if timings:
        st.caption(
            f"⏱ retrieval {timings['retrieval']} ms · generation "
            f"{timings['generation']} ms · total {timings['total'] / 1000:.1f} s"
        )


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📂 Documents")
    online = backend_ok()
    if not online:
        st.error("Backend not running. Start it with: python launcher.py")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Upload & index", disabled=not online):
        for uf in files:
            with st.spinner(f"Indexing {uf.name}…"):
                r = requests.post(
                    f"{API}/upload",
                    files={"file": (uf.name, uf.getvalue(), "application/pdf")},
                    timeout=300,
                )
            if r.ok:
                d = r.json()
                st.success(f"{d['file']}: {d['pages']} pages → {d['chunks']} chunks")
            else:
                st.error(f"{uf.name}: {r.json().get('detail', r.text)}")

    if online:
        docs = requests.get(f"{API}/documents", timeout=10).json()["documents"]
        if docs:
            st.subheader("Indexed")
            for d in docs:
                st.write(f"• {d['file']} ({d['chunks']} chunks)")
            if st.button("Clear all documents"):
                requests.delete(f"{API}/documents", timeout=30)
                st.session_state.messages = []
                st.rerun()

    st.divider()
    top_k = st.slider("Passages to retrieve (k)", 1, 10, 4)

# --------------------------------------------------------------------- chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"), m.get("timings"))

question = st.chat_input("Ask something about your documents…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources, timings = [], None
        try:
            with st.spinner("Searching your documents…"):
                r = requests.get(f"{API}/ask", params={"q": question, "k": top_k}, timeout=180)
            if r.ok:
                data = r.json()
                answer, sources, timings = data["answer"], data["sources"], data["timings_ms"]
            else:
                answer = f"⚠️ {r.json().get('detail', r.text)}"
        except requests.RequestException as e:
            answer = f"⚠️ Backend request failed: {e}"
        st.markdown(answer)
        show_sources(sources, timings)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "timings": timings}
    )
