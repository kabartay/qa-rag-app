from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG Application for QA with PDF Docs", layout="wide")

readme_path = Path(__file__).parent.parent / "README.md"
if readme_path.exists():
    readme = readme_path.read_text(encoding="utf-8")
    st.markdown(readme, unsafe_allow_html=True)
else:
    st.error("README.md not found.")
