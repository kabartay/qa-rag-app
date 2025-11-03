"""
Unified RAG Evaluator.
Compare Simple vs Enhanced RAG.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, cast

import anthropic
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is in Python path (so "apps" imports work)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load environment variables
load_dotenv()

# Import both RAG systems
try:
    from apps.rag_app import SimpleRAG
    from apps.rag_app_enhanced import EnhancedRAG

    BOTH_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    BOTH_AVAILABLE = False


class RAGEvaluator:
    """Comprehensive RAG evaluation with comparison support"""

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Evaluate with 4 metrics using LLM-as-judge"""

        eval_prompt = f"""You are an expert evaluator for RAG systems. Evaluate the following:

**Question:** {question}

**Generated Answer:** {generated_answer}

**Ground Truth Answer:** {ground_truth}

**Retrieved Context:** {context[:1000] if context else "Not provided"}

Evaluate on these dimensions (score 1-5, where 5 is best):

1. **Faithfulness** (1-5): Is the generated answer supported by the retrieved context?
   - 5: Fully supported, no hallucinations
   - 3: Mostly supported with minor issues
   - 1: Contains unsupported claims or hallucinations

2. **Answer Relevancy** (1-5): Does the answer directly address the question?
   - 5: Perfectly addresses all aspects
   - 3: Addresses main point but misses details
   - 1: Off-topic or irrelevant

3. **Context Relevancy** (1-5): Is the retrieved context relevant to answering the question?
   - 5: Highly relevant, contains all needed info
   - 3: Somewhat relevant but incomplete
   - 1: Irrelevant or missing key information

4. **Correctness** (1-5): How accurate is the answer compared to ground truth?
   - 5: Semantically equivalent, all key points covered
   - 3: Partially correct, some missing information
   - 1: Incorrect or contradictory

Respond in JSON format:
{{
  "faithfulness": <score>,
  "answer_relevancy": <score>,
  "context_relevancy": <score>,
  "correctness": <score>,
  "explanation": "Brief explanation",
  "key_issues": []
}}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": eval_prompt}],
            )

            response = message.content[0].text
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return cast(dict[str, Any], parsed)
            else:
                return self._default_scores()
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return self._default_scores()

    def _default_scores(self) -> dict[str, Any]:
        """Return default zeroed evaluation metrics."""
        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_relevancy": 0,
            "correctness": 0,
            "explanation": "Evaluation failed",
            "key_issues": [],
        }

    def batch_evaluate(
        self,
        rag_system: Any,
        qa_pairs: list[dict[str, Any]],
        progress_bar: Any,
        status_text: Any,
        system_name: str,
    ) -> dict[str, Any]:
        """Evaluate a RAG system on multiple Q&A pairs"""
        results: list[dict[str, Any]] = []

        for i, qa in enumerate(qa_pairs):
            status_text.text(
                f"[{system_name}] Evaluating {i+1}/{len(qa_pairs)}: {qa['question'][:40]}..."
            )

            try:
                rag_response = rag_system.answer_question(qa["question"])

                scores = self.evaluate_answer(
                    question=qa["question"],
                    generated_answer=rag_response["answer"],
                    ground_truth=qa["answer"],
                    context=rag_response.get("context_used", ""),
                )

                results.append(
                    {
                        "question": qa["question"],
                        "ground_truth": qa["answer"],
                        "generated_answer": rag_response["answer"],
                        "scores": scores,
                        "difficulty": qa.get("difficulty", "unknown"),
                        "type": qa.get("type", "unknown"),
                    }
                )
            except Exception as e:
                st.warning(f"Skipped question {i+1}: {str(e)}")

            progress_bar.progress((i + 1) / len(qa_pairs))

            # Rate limiting: 6 seconds between evaluations
            if i < len(qa_pairs) - 1:
                time.sleep(6)

        status_text.text(f"{system_name} evaluation complete!")
        return self._aggregate_results(results, system_name)

    def _aggregate_results(self, results: list[dict[str, Any]], system_name: str) -> dict[str, Any]:
        """Calculate aggregate metrics"""
        if not results:
            return {}

        metrics = ["faithfulness", "answer_relevancy", "context_relevancy", "correctness"]

        aggregate: dict[str, Any] = {
            "system_name": system_name,
            "total_questions": len(results),
            "timestamp": datetime.now().isoformat(),
            "individual_results": results,
        }

        # Overall averages
        for metric in metrics:
            scores = [r["scores"][metric] for r in results if r["scores"][metric] > 0]
            aggregate[f"avg_{metric}"] = sum(scores) / len(scores) if scores else 0

        # By difficulty
        difficulties = {r["difficulty"] for r in results}
        aggregate["by_difficulty"] = {}

        for diff in difficulties:
            diff_results = [r for r in results if r["difficulty"] == diff]
            if diff_results:
                aggregate["by_difficulty"][diff] = {
                    "count": len(diff_results),
                    "avg_correctness": sum(r["scores"]["correctness"] for r in diff_results)
                    / len(diff_results),
                }

        # By question type
        types = {r["type"] for r in results}
        aggregate["by_type"] = {}

        for qtype in types:
            type_results = [r for r in results if r["type"] == qtype]
            if type_results:
                aggregate["by_type"][qtype] = {
                    "count": len(type_results),
                    "avg_correctness": sum(r["scores"]["correctness"] for r in type_results)
                    / len(type_results),
                }

        return aggregate


def main() -> None:
    st.set_page_config(page_title="Unified RAG Evaluator", page_icon="⚖️", layout="wide")

    st.title("Unified RAG System Evaluator")
    st.markdown("**Compare Simple RAG vs Enhanced RAG side-by-side**")

    # Sidebar
    with st.sidebar:
        st.header("API Keys")

        anthropic_key = st.text_input(
            "Anthropic", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password"
        )
        pinecone_key = st.text_input(
            "Pinecone", value=os.getenv("PINECONE_API_KEY", ""), type="password"
        )
        cohere_key = st.text_input("Cohere", value=os.getenv("COHERE_API_KEY", ""), type="password")

        st.markdown("---")
        st.header("Evaluation Mode")

        eval_mode = st.radio(
            "Select system(s) to evaluate:",
            ["Simple RAG Only", "Enhanced RAG Only", "Both (Comparison)"],
            index=2,
        )

        st.markdown("---")
        st.info(f"**Selected:** {eval_mode}")
        if eval_mode == "Both (Comparison)":
            st.success("✨ Will compare both systems on same questions")

    # Session state
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "simple_results" not in st.session_state:
        st.session_state.simple_results = None
    if "enhanced_results" not in st.session_state:
        st.session_state.enhanced_results = None

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📁 Load Data", "🧪 Run Evaluation", "📈 Results"])

    with tab1:
        st.header("Load Document & Q&A Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Document")
            uploaded_doc = st.file_uploader("Upload PDF", type=["pdf"])
            doc_text = st.text_area("Or paste text", height=150)

            if "document_text" not in st.session_state:
                st.session_state.document_text = ""

            if uploaded_doc:
                try:
                    from io import BytesIO

                    import PyPDF2

                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_doc.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n\n"
                    st.session_state.document_text = text
                    st.success(f"PDF loaded ({len(text)} chars)")
                except Exception as e:
                    st.error(f"PDF error: {str(e)}")
            elif doc_text:
                st.session_state.document_text = doc_text

        with col2:
            st.subheader("Q&A Dataset")
            uploaded_qa = st.file_uploader("Upload Q&A JSON", type=["json"])

            if uploaded_qa:
                try:
                    qa_data = json.load(uploaded_qa)
                    st.session_state.qa_pairs = qa_data
                    st.success(f"Loaded {len(qa_data)} Q&A pairs")

                    with st.expander("Preview"):
                        for qa in qa_data[:3]:
                            st.markdown(f"**Q:** {qa['question'][:60]}...")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab2:
        st.header("Run Evaluation")

        if not st.session_state.document_text:
            st.warning("Please load document first (Tab 1)")
        elif not st.session_state.qa_pairs:
            st.warning("Please load Q&A dataset first (Tab 1)")
        else:
            st.success(f"Ready: {len(st.session_state.qa_pairs)} questions loaded")

            num_questions = st.slider(
                "Questions to evaluate",
                1,
                len(st.session_state.qa_pairs),
                min(10, len(st.session_state.qa_pairs)),
            )

            st.info(f"Estimated time: ~{num_questions}min (6 sec per question with rate limiting)")

            if st.button("Start Evaluation", type="primary"):
                if not anthropic_key:
                    st.error("Anthropic API key required")
                    return

                evaluator = RAGEvaluator(anthropic_key)
                selected_questions = st.session_state.qa_pairs[:num_questions]

                # Evaluate Simple RAG
                if eval_mode in ["Simple RAG Only", "Both (Comparison)"]:
                    st.markdown("### Evaluating Simple RAG...")
                    progress_simple = st.progress(0)
                    status_simple = st.empty()

                    simple_rag = SimpleRAG(anthropic_key)
                    simple_rag.load_document(st.session_state.document_text)

                    st.session_state.simple_results = evaluator.batch_evaluate(
                        simple_rag, selected_questions, progress_simple, status_simple, "Simple RAG"
                    )

                # Evaluate Enhanced RAG
                if eval_mode in ["Enhanced RAG Only", "Both (Comparison)"]:
                    if not all([pinecone_key, cohere_key]):
                        st.error("Pinecone + Cohere keys needed for Enhanced RAG")
                        return

                    st.markdown("### Evaluating Enhanced RAG...")
                    progress_enhanced = st.progress(0)
                    status_enhanced = st.empty()

                    enhanced_rag = EnhancedRAG(anthropic_key, pinecone_key, cohere_key)
                    enhanced_rag.load_document(st.session_state.document_text, "eval_doc")

                    st.session_state.enhanced_results = evaluator.batch_evaluate(
                        enhanced_rag,
                        selected_questions,
                        progress_enhanced,
                        status_enhanced,
                        "Enhanced RAG",
                    )

                st.success("Evaluation complete!")
                st.balloons()

    with tab3:
        st.header("Evaluation Results")

        simple_res = st.session_state.simple_results
        enhanced_res = st.session_state.enhanced_results

        if not simple_res and not enhanced_res:
            st.info("Run evaluation first")
            return

        # Comparison mode
        if simple_res and enhanced_res:
            st.subheader("Side-by-Side Comparison")

            col1, col2, col3 = st.columns([5, 5, 2])

            with col1:
                st.markdown("### Simple")
            with col2:
                st.markdown("### Enhanced")
            with col3:
                st.markdown("### 🏆 Winner")

            st.markdown("---")

            # Overall scores
            metrics = ["faithfulness", "answer_relevancy", "context_relevancy", "correctness"]

            for metric in metrics:
                col1, col2, col3 = st.columns([5, 5, 2])

                simple_score = simple_res.get(f"avg_{metric}", 0)
                enhanced_score = enhanced_res.get(f"avg_{metric}", 0)

                with col1:
                    st.metric(metric.replace("_", " ").title(), f"{simple_score:.2f}/5")
                with col2:
                    st.metric(metric.replace("_", " ").title(), f"{enhanced_score:.2f}/5")
                with col3:
                    if enhanced_score > simple_score + 0.1:
                        st.markdown("Enhanced")
                    elif simple_score > enhanced_score + 0.1:
                        st.markdown("Simple")
                    else:
                        st.markdown("🤝 Tie")

            # Overall average
            st.markdown("---")

            simple_avg = sum(simple_res.get(f"avg_{m}", 0) for m in metrics) / 4
            enhanced_avg = sum(enhanced_res.get(f"avg_{m}", 0) for m in metrics) / 4

            col1, col2, col3 = st.columns([5, 5, 2])
            with col1:
                st.metric("**OVERALL**", f"{simple_avg:.2f}/5")
            with col2:
                st.metric("**OVERALL**", f"{enhanced_avg:.2f}/5")
            with col3:
                if enhanced_avg > simple_avg + 0.2:
                    st.markdown("### 🏆")
                    st.success("Enhanced!")
                elif simple_avg > enhanced_avg + 0.2:
                    st.markdown("### 🏆")
                    st.success("Simple!")
                else:
                    st.markdown("### 🤝")
                    st.info("Similar")

            # Recommendation
            st.markdown("---")
            st.subheader("💡 Recommendation")

            diff = enhanced_avg - simple_avg
            if diff > 0.3:
                st.success(
                    f"**Use Enhanced RAG** - {diff:.2f} points better ({(diff/simple_avg)*100:.1f}% improvement)"
                )
            elif diff > 0.1:
                st.info(f"**Consider Enhanced RAG** - {diff:.2f} points better")
            elif diff < -0.1:
                st.warning("**Simple RAG performs better** - Unexpected! Check Enhanced RAG setup.")
            else:
                st.info("**Both similar** - Simple RAG is sufficient (simpler + cheaper)")

            # Export both
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Simple RAG Results",
                    json.dumps(simple_res, indent=2),
                    "simple_rag_results.json",
                    "application/json",
                )
            with col2:
                st.download_button(
                    "📥 Download Enhanced RAG Results",
                    json.dumps(enhanced_res, indent=2),
                    "enhanced_rag_results.json",
                    "application/json",
                )

        # Single system mode
        else:
            results = simple_res or enhanced_res
            system_name = results.get("system_name", "RAG System")

            st.subheader(f"{system_name} Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Faithfulness", f"{results.get('avg_faithfulness', 0):.2f}/5")
            with col2:
                st.metric("Answer Relevancy", f"{results.get('avg_answer_relevancy', 0):.2f}/5")
            with col3:
                st.metric("Context Relevancy", f"{results.get('avg_context_relevancy', 0):.2f}/5")
            with col4:
                st.metric("Correctness", f"{results.get('avg_correctness', 0):.2f}/5")

            avg_overall = (
                results.get("avg_faithfulness", 0)
                + results.get("avg_answer_relevancy", 0)
                + results.get("avg_context_relevancy", 0)
                + results.get("avg_correctness", 0)
            ) / 4

            st.markdown(f"### Overall Score: **{avg_overall:.2f}/5**")

            # Export
            st.download_button(
                f"📥 Download {system_name} Results",
                json.dumps(results, indent=2),
                f"{system_name.lower().replace(' ', '_')}_results.json",
                "application/json",
            )


if __name__ == "__main__":
    main()
