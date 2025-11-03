from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG Application Documentation", layout="wide")

st.title("RAG Application for QA with PDF Docs")

readme_path = Path(__file__).resolve().parents[1] / "README.md"

if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
else:
    st.warning("README.md not found. Make sure it's copied into the Docker image.")
