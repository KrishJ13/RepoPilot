"""Streamlit reruns this script on interaction; a form waits for Analyse."""

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RepoPilot V1", page_icon="📖")
st.title("RepoPilot")
st.caption("Understand a public repository from its README. V0: direct prompting.")
with st.form("analyse"):
    url = st.text_input("Public GitHub repository", "https://github.com/pallets/flask")
    submitted = st.form_submit_button("Analyse")
if submitted:
    try:
        with st.spinner("Fetching README and asking the model…"):
            # The container uses api:8000; local development uses localhost:8000.
            response = httpx.post(
                os.getenv("API_URL", "http://localhost:8000") + "/api/v1/repositories/summary",
                json={"repository_url": url}, timeout=120,
            )
        if response.is_error:
            st.error(str(response.json().get("detail", "Analysis failed.")))
        else:
            result = response.json()
            summary = result["summary"]
            st.subheader(summary["name"])
            st.write(summary["purpose"])
            for field, label in [("main_features", "Main features"), ("technologies", "Technologies"), ("limitations", "Limitations")]:
                st.subheader(label)
                for item in summary[field]:
                    st.write("• " + item)
            st.subheader("How to run")
            st.text(summary["how_to_run"])
            st.caption("Source: " + str(result["readme_url"] or result["repository_url"]))
            st.caption(f"{result['model']} · {result['elapsed_seconds']}s · {result['input_tokens']} input / {result['output_tokens']} output tokens")
            with st.expander("Inspect the API response"):
                st.json(result)
    except httpx.RequestError:
        st.error("Cannot reach the API. Start it on port 8000 and try again.")
st.caption("Generated summaries can be wrong. Check the README before following advice.")


st.divider()
st.header("V1 · Repository Q&A")
st.write("Index documentation once, then ask questions and inspect the evidence.")
rag_url = st.text_input("Repository to index and ask", "https://github.com/encode/httpx", key="rag_url")
base_url = os.getenv("API_URL", "http://localhost:8000")
if st.button("Index repository"):
    try:
        with st.spinner("Fetching sources, embedding chunks and writing Qdrant… First use downloads the embedding model."):
            indexed = httpx.post(base_url + "/api/v1/repositories/index", json={"repository_url": rag_url}, timeout=600)
        if indexed.is_error:
            st.error(str(indexed.json().get("detail")))
        else:
            info = indexed.json()
            st.success(f"Indexed {info['documents']} documents into {info['chunks']} chunks for {info['repository']}.")
            for warning in info["warnings"]:
                st.caption(warning)
    except httpx.RequestError:
        st.error("Index request failed or timed out. Check the API and Qdrant logs.")
with st.form("ask"):
    question = st.text_input("Question", "How do I install HTTPX?")
    top_k = st.slider("Evidence chunks (K)", 1, 10, 5)
    kind = st.selectbox("Source type", ["all", "readme", "documentation", "release", "issue", "metadata"])
    asked = st.form_submit_button("Ask")
if asked:
    try:
        with st.spinner("Retrieving evidence and generating a cited answer…"):
            response = httpx.post(base_url + "/api/v1/repositories/ask", json={
                "repository_url": rag_url, "question": question, "top_k": top_k,
                "document_type": None if kind == "all" else kind,
            }, timeout=120)
        if response.is_error:
            st.error(str(response.json().get("detail")))
        else:
            result = response.json()
            sources = {item["chunk"]["chunk_id"]: item["chunk"] for item in result["evidence"]}
            for claim in result["answer"]["claims"]:
                st.write(claim["text"])
                for chunk_id in claim["chunk_ids"]:
                    source = sources[chunk_id]
                    # Source addresses come from stored GitHub evidence, not model output.
                    st.link_button(source["title"] + " · " + chunk_id[:8], source["source_url"])
            if result["answer"]["insufficient_evidence"]:
                st.warning("Evidence is insufficient for a complete answer.")
                st.write(result["answer"]["missing_information"])
            with st.expander("Retrieved evidence and similarity scores"):
                for item in result["evidence"]:
                    st.caption(f"{item['chunk']['title']} · cosine {item['score']:.3f}")
                    st.text(item["chunk"]["text"])
            st.caption(f"{result['model']} · {result['elapsed_seconds']}s · {result['input_tokens']} input / {result['output_tokens']} output tokens")
    except httpx.RequestError:
        st.error("Cannot reach the API, or the answer timed out.")
