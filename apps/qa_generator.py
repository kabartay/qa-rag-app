"""
Groundtruth Q&A Pair Generator with PDF Upload Support.
Automatically generates question-answer pairs from documents for evaluation.
"""

import csv
import json
import os
from io import BytesIO, StringIO
from typing import Any, cast

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Load environment variables
load_dotenv()


class QAGenerator:
    """Generates diverse, high-quality Q&A pairs from documents"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_qa_pairs(
        self, text: str, num_pairs: int = 10, difficulty: str = "mixed"
    ) -> list[dict[Any, Any]]:
        """Generate Q&A pairs from document text"""

        prompt = f"""You are an expert at creating evaluation datasets for RAG systems.

Given the following document, generate {num_pairs} diverse question-answer pairs that would be useful for evaluating a RAG system's performance.

Document:
{text[:4000]}  # Truncate for context limits

Requirements:
1. Generate questions at {difficulty} difficulty levels (easy factual, medium analytical, hard inference)
2. Each question should be answerable from the document
3. Include diverse question types:
   - Factual retrieval (What, When, Who)
   - Conceptual understanding (Why, How)
   - Comparative (Difference between X and Y)
   - Numerical/quantitative
   - Multi-hop reasoning (requiring multiple pieces of info)
4. Answers should be comprehensive but concise (2-4 sentences)
5. Include the specific section/context where the answer can be found

Format your response as a JSON array:
[
  {{
    "id": 1,
    "question": "...",
    "answer": "...",
    "difficulty": "easy|medium|hard",
    "type": "factual|conceptual|comparative|numerical|multi-hop",
    "relevant_context": "brief excerpt from document"
  }},
  ...
]

Generate exactly {num_pairs} Q&A pairs."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text

            # Extract JSON from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                qa_pairs = cast(list[dict[str, Any]], json.loads(json_str))
                return qa_pairs
            else:
                st.error("Could not parse JSON response")
                return []

        except Exception as e:
            st.error(f"Error generating Q&A pairs: {str(e)}")
            return []


def export_to_json(qa_pairs: list[dict]) -> str:
    """Export Q&A pairs to JSON format"""
    return json.dumps(qa_pairs, indent=2, ensure_ascii=False)


def export_to_csv(qa_pairs: list[dict]) -> str:
    """Export Q&A pairs to CSV format"""
    output = StringIO()

    if not qa_pairs:
        return ""

    fieldnames = ["id", "question", "answer", "difficulty", "type", "relevant_context"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    for pair in qa_pairs:
        writer.writerow({k: pair.get(k, "") for k in fieldnames})

    return output.getvalue()


def extract_text_from_pdf(uploaded_file: UploadedFile) -> str:
    """Extract text from uploaded PDF file"""
    try:
        import PyPDF2

        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"

        return text
    except Exception as e:
        raise Exception(f"Error extracting PDF: {str(e)}") from e


def main() -> None:
    st.set_page_config(page_title="Q&A Groundtruth Generator", page_icon="@", layout="wide")

    st.title("Groundtruth Q&A Dataset Generator")
    st.markdown("Automatically generate evaluation datasets from documents for RAG system testing.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Get API key from environment or user input
        default_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("Anthropic API Key", value=default_api_key, type="password")

        st.markdown("---")
        st.header("Generation Settings")

        num_pairs = st.slider("Number of Q&A pairs", 5, 50, 20)
        difficulty = st.selectbox("Difficulty distribution", ["easy", "medium", "hard", "mixed"])

        st.markdown("---")
        st.markdown("### Question Types")
        st.markdown("- **Factual**: Direct information retrieval")
        st.markdown("- **Conceptual**: Understanding principles")
        st.markdown("- **Comparative**: Comparing concepts")
        st.markdown("- **Numerical**: Quantitative questions")
        st.markdown("- **Multi-hop**: Requires connecting info")

    # Initialize generator
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []

    # Document input
    st.header("1 Input Document")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])

    document_text = ""

    with tab1:
        st.markdown("**Upload a PDF file**")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload")

        if uploaded_file:
            try:
                with st.spinner("Extracting text from PDF..."):
                    document_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"Extracted {len(document_text)} characters from PDF")

                    # Show preview
                    with st.expander("Preview extracted text"):
                        st.text(
                            document_text[:1000] + "..."
                            if len(document_text) > 1000
                            else document_text
                        )
            except Exception as e:
                st.error(f"{str(e)}")
                st.info("Try the 'Paste Text' tab instead")

    with tab2:
        st.markdown("**Paste document text directly**")
        pasted_text = st.text_area(
            "Document text",
            height=300,
            help="Paste the text content of your document",
            key="text_paste",
        )
        if pasted_text:
            document_text = pasted_text
            st.info(f"{len(document_text)} characters entered")

    # Tips
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("** Tips for better results:**")
        st.markdown("- Use complete sections with context")
        st.markdown("- Include 500-5000 words")
        st.markdown("- Technical docs work best")
        st.markdown("- Multiple topics = diverse questions")

    # Generation controls
    st.header("2 Generate Q&A Pairs")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate Dataset", type="primary", disabled=not document_text or not api_key):
            if not api_key:
                st.error("Please provide an Anthropic API key in the sidebar")
            elif not document_text:
                st.error("Please upload a PDF or paste text")
            else:
                generator = QAGenerator(api_key)

                with st.spinner(f"Generating {num_pairs} Q&A pairs..."):
                    qa_pairs = generator.generate_qa_pairs(
                        document_text, num_pairs=num_pairs, difficulty=difficulty
                    )

                    if qa_pairs:
                        st.session_state.qa_pairs = qa_pairs
                        st.success(f"Generated {len(qa_pairs)} Q&A pairs!")
                        st.balloons()
                    else:
                        st.error("Failed to generate Q&A pairs")

    with col2:
        if st.button("Generate More", disabled=not st.session_state.qa_pairs or not document_text):
            if document_text and api_key:
                generator = QAGenerator(api_key)
                with st.spinner("Generating additional pairs..."):
                    new_pairs = generator.generate_qa_pairs(
                        document_text, num_pairs=10, difficulty=difficulty
                    )
                    if new_pairs:
                        # Update IDs
                        max_id = max([p["id"] for p in st.session_state.qa_pairs])
                        for i, pair in enumerate(new_pairs):
                            pair["id"] = max_id + i + 1
                        st.session_state.qa_pairs.extend(new_pairs)
                        st.success(f"Added {len(new_pairs)} more pairs!")

    with col3:
        if st.button("Clear All", disabled=not st.session_state.qa_pairs):
            st.session_state.qa_pairs = []
            st.rerun()

    # Display results
    if st.session_state.qa_pairs:
        st.header("3 Generated Q&A Pairs")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        pairs = st.session_state.qa_pairs
        difficulties = [p.get("difficulty", "unknown") for p in pairs]

        with col1:
            st.metric("Total Pairs", len(pairs))
        with col2:
            st.metric("Easy", difficulties.count("easy"))
        with col3:
            st.metric("Medium", difficulties.count("medium"))
        with col4:
            st.metric("Hard", difficulties.count("hard"))

        # Display pairs
        for i, pair in enumerate(pairs):
            with st.expander(f"Q{pair.get('id', i+1)}: {pair.get('question', '')[:100]}..."):
                st.markdown(f"**Question:** {pair.get('question', '')}")
                st.markdown(f"**Answer:** {pair.get('answer', '')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Difficulty: {pair.get('difficulty', 'N/A')}")
                with col2:
                    st.caption(f"Type: {pair.get('type', 'N/A')}")

                if pair.get("relevant_context"):
                    st.markdown("**Relevant Context:**")
                    st.text(pair.get("relevant_context", "")[:300] + "...")

        # Export section
        st.header("4 Export Dataset")

        col1, col2 = st.columns(2)

        with col1:
            json_data = export_to_json(pairs)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="qa_groundtruth.json",
                mime="application/json",
                help="📥 Download as JSON for evaluation",
            )

        with col2:
            csv_data = export_to_csv(pairs)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="qa_groundtruth.csv",
                mime="text/csv",
                help="📥 Download as CSV for spreadsheet analysis",
            )


if __name__ == "__main__":
    main()
