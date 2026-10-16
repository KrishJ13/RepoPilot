"""Streamlit reruns this script on interaction; a form waits for Analyse."""

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RepoPilot V0", page_icon="📖")
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
