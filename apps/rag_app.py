"""
Simple RAG without Vector DB or re-ranking, just using Claude API.
Handles small documents well.
"""

import os
from typing import Any

import anthropic
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAG Application
st.set_page_config(page_title="Simple RAG System", page_icon="🔹", layout="wide")


class SimpleRAG:
    """Simple RAG implementation using Claude's long context"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.document_content = ""
        self.chunks: list[dict[str, Any]] = []

    def load_document(self, text: str) -> bool:
        """Load and chunk document"""
        self.document_content = text
        # Simple chunking by paragraphs/sections
        self.chunks = self._chunk_text(text)
        return True

    def _chunk_text(
        self, text: str, chunk_size: int = 2000, overlap: int = 200
    ) -> list[dict[str, Any]]:
        """Chunk text with overlap for better context"""
        chunks: list[dict[str, Any]] = []
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"id": len(chunks), "text": chunk_text, "start_idx": i})

        return chunks

    def answer_question(self, question: str) -> dict:
        """Answer question using RAG approach"""

        # Use all chunks (long context approach)
        context = "\n\n---\n\n".join([chunk["text"] for chunk in self.chunks[:10]])

        # Create prompt for Claude
        prompt = f"""You are an expert assistant helping users understand a technical document.

Context from the document:
{context}

User question: {question}

Please answer the question based on the provided context. If the context doesn't contain enough information to fully answer the question, say so clearly. Always cite specific sections when possible.

Answer in the same language as the question (French or English)."""

        # Call Claude API
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = message.content[0].text

            return {
                "answer": answer,
                "context_used": context[:500] + "..." if len(context) > 500 else context,
                "chunks_used": len(self.chunks[:10]),
            }

        except Exception as e:
            return {"answer": f"Error: {str(e)}", "context_used": "", "chunks_used": 0}


def main() -> None:
    st.title("Simple RAG-based Question-Answering System")
    st.markdown("Upload a PDF document and ask questions about its content.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("API Key")

        # Get API key from environment or user input
        default_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_key = st.text_input(
            "Anthropic API Key",
            value=default_api_key,
            type="password",
            help="Enter your Anthropic API key",
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("This RAG system uses:")
        st.markdown("- **Retrieval**: Chunk-based document retrieval")
        st.markdown("- **Generation**: Claude Sonnet 4 for answers")
        st.markdown("- **Context**: Long context window for accuracy")

    # Initialize RAG system
    if "rag" not in st.session_state:
        st.session_state.rag = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Document upload section
    st.header("1 Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf", "txt"])

    # For demo purposes, also accept direct text input
    st.markdown("**Or paste document text directly:**")
    text_input = st.text_area("Document text", height=150)

    if st.button("Load Document", type="primary"):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        else:
            document_text = ""

            if uploaded_file:
                # Read file content
                if uploaded_file.type == "application/pdf":
                    try:
                        from io import BytesIO

                        import PyPDF2

                        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                        document_text = ""
                        for page in pdf_reader.pages:
                            document_text += page.extract_text() + "\n\n"
                    except Exception as e:
                        st.error(f"Error reading PDF: {str(e)}")
                        st.info("Try pasting the text directly instead.")
                else:
                    document_text = uploaded_file.read().decode()
            elif text_input:
                document_text = text_input

            if document_text:
                with st.spinner("Loading document..."):
                    st.session_state.rag = SimpleRAG(api_key)
                    st.session_state.rag.load_document(document_text)
                st.success(f"Document loaded! ({len(st.session_state.rag.chunks)} chunks created)")
            else:
                st.error("Please provide a document (upload PDF or paste text).")

    # Question-answering section
    st.header("2 Ask Questions")

    if st.session_state.rag is None:
        st.info("Please load a document first.")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "context" in message:
                    with st.expander("📄 Context used"):
                        st.text(message["context"])

        # Chat input
        if question := st.chat_input("Ask a question about the document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag.answer_question(question)

                    st.markdown(result["answer"])

                    with st.expander("📄 Context used"):
                        st.text(result["context_used"])
                        st.caption(f"Used {result['chunks_used']} document chunks")

                    # Add assistant message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "context": result["context_used"],
                        }
                    )

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
