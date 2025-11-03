from pathlib import Path

import streamlit as st

st.title("RAG Application for QA with PDF docs")

readme_path = Path(__file__).resolve().parents[1] / "README.md"
if readme_path.exists():
    st.markdown(readme_path.read_text())
else:
    st.warning("README.md not found.")
