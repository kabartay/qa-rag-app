import streamlit as st
import anthropic
import json
from typing import List, Dict
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# Add apps directory to path so we can import SimpleRAG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import SimpleRAG from rag_app
try:
    from apps.rag_app import SimpleRAG
except ImportError:
    st.error("❌ Could not import SimpleRAG. Make sure apps/rag_app.py exists and is correct.")
    st.stop()

# Automated RAG Evaluator - Connects to Simple RAG
# Evaluates RAG system quality using groundtruth Q&A pairs

class RAGEvaluator:
    """Comprehensive RAG evaluation using multiple metrics"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def evaluate_answer(self, question: str, generated_answer: str, 
                       ground_truth: str, context: str = "") -> Dict:
        """
        Evaluate a single answer using multiple metrics:
        1. Faithfulness (answer supported by context)
        2. Answer Relevancy (answer addresses question)
        3. Context Relevancy (retrieved context is relevant)
        4. Correctness (compared to ground truth)
        """
        
        # Use Claude as a judge (LLM-as-judge pattern)
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
  "explanation": "Brief explanation of scores",
  "key_issues": ["list", "of", "issues"] or []
}}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            
            response = message.content[0].text
            
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                scores = json.loads(response[json_start:json_end])
                return scores
            else:
                return self._default_scores()
                
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return self._default_scores()
    
    def _default_scores(self):
        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_relevancy": 0,
            "correctness": 0,
            "explanation": "Evaluation failed",
            "key_issues": ["Evaluation error"]
        }
    
    def batch_evaluate(self, rag_system: SimpleRAG, qa_pairs: List[Dict], 
                      progress_bar, status_text) -> Dict:
        """Evaluate RAG system on multiple Q&A pairs"""
        results = []
        
        for i, qa in enumerate(qa_pairs):
            status_text.text(f"Evaluating {i+1}/{len(qa_pairs)}: {qa['question'][:50]}...")
            
            # Get RAG system answer
            rag_response = rag_system.answer_question(qa['question'])
            
            # Evaluate
            scores = self.evaluate_answer(
                question=qa['question'],
                generated_answer=rag_response['answer'],
                ground_truth=qa['answer'],
                context=rag_response.get('context_used', '')
            )
            
            results.append({
                'question': qa['question'],
                'ground_truth': qa['answer'],
                'generated_answer': rag_response['answer'],
                'scores': scores,
                'difficulty': qa.get('difficulty', 'unknown'),
                'type': qa.get('type', 'unknown')
            })
            
            progress_bar.progress((i + 1) / len(qa_pairs))
            time.sleep(0.5)  # Rate limiting
        
        status_text.text("✅ Evaluation complete!")
        
        # Aggregate metrics
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics"""
        
        if not results:
            return {}
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy', 'correctness']
        
        aggregate = {
            'total_questions': len(results),
            'timestamp': datetime.now().isoformat(),
            'individual_results': results
        }
        
        # Overall averages
        for metric in metrics:
            scores = [r['scores'][metric] for r in results if metric in r['scores']]
            aggregate[f'avg_{metric}'] = sum(scores) / len(scores) if scores else 0
        
        # By difficulty
        difficulties = set(r['difficulty'] for r in results)
        aggregate['by_difficulty'] = {}
        
        for diff in difficulties:
            diff_results = [r for r in results if r['difficulty'] == diff]
            if diff_results:
                aggregate['by_difficulty'][diff] = {
                    'count': len(diff_results),
                    'avg_correctness': sum(r['scores']['correctness'] for r in diff_results) / len(diff_results)
                }
        
        # By question type
        types = set(r['type'] for r in results)
        aggregate['by_type'] = {}
        
        for qtype in types:
            type_results = [r for r in results if r['type'] == qtype]
            if type_results:
                aggregate['by_type'][qtype] = {
                    'count': len(type_results),
                    'avg_correctness': sum(r['scores']['correctness'] for r in type_results) / len(type_results)
                }
        
        return aggregate


def main():
    st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")
    
    st.title("📊 Automated RAG System Evaluator")
    st.markdown("Comprehensive evaluation of RAG systems using groundtruth datasets")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API key from environment or user input
        default_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_key = st.text_input(
            "Anthropic API Key",
            value=default_api_key,
            type="password"
        )
        
        st.markdown("---")
        st.header("Evaluation Metrics")
        st.markdown("**Faithfulness**: Answer supported by context")
        st.markdown("**Answer Relevancy**: Answer addresses question")
        st.markdown("**Context Relevancy**: Retrieved context is relevant")
        st.markdown("**Correctness**: Compared to ground truth")
        
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("1. Loads your document into Simple RAG")
        st.markdown("2. Asks each question from groundtruth")
        st.markdown("3. Evaluates answers with 4 metrics")
        st.markdown("4. Shows aggregate results")
    
    # Initialize session state
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    
    # Tabs for workflow
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Load Document", "📋 Load Q&A Dataset", "🧪 Run Evaluation", "📈 Results"])
    
    with tab1:
        st.header("Load Document into RAG System")
        
        st.markdown("**Upload PDF or paste text:**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_doc = st.file_uploader("Upload PDF", type=['pdf'], key="doc_upload")
            doc_text = st.text_area("Or paste document text", height=200)
        
        with col2:
            st.info("💡 This is the document your RAG will be tested on")
        
        if st.button("🚀 Load Document into RAG", type="primary"):
            if not api_key:
                st.error("⚠️ Please provide API key")
            else:
                document_text = ""
                
                if uploaded_doc:
                    try:
                        import PyPDF2
                        from io import BytesIO
                        
                        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_doc.read()))
                        for page in pdf_reader.pages:
                            document_text += page.extract_text() + "\n\n"
                    except Exception as e:
                        st.error(f"PDF read error: {str(e)}")
                elif doc_text:
                    document_text = doc_text
                
                if document_text:
                    with st.spinner("Loading document into RAG system..."):
                        st.session_state.rag_system = SimpleRAG(api_key)
                        st.session_state.rag_system.load_document(document_text)
                        st.session_state.document_loaded = True
                    
                    st.success(f"✅ Document loaded! ({len(st.session_state.rag_system.chunks)} chunks)")
                else:
                    st.error("⚠️ Please provide a document")
    
    with tab2:
        st.header("Load Groundtruth Q&A Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_qa = st.file_uploader("Upload Q&A JSON file", type=['json'], key="qa_upload")
            
            if uploaded_qa:
                try:
                    qa_data = json.load(uploaded_qa)
                    st.session_state.qa_pairs = qa_data
                    st.success(f"✅ Loaded {len(qa_data)} Q&A pairs")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        with col2:
            st.markdown("**JSON Format:**")
            st.code('''[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "difficulty": "easy",
    "type": "factual"
  }
]''', language='json')
        
        if st.session_state.qa_pairs:
            st.markdown(f"**Dataset loaded:** {len(st.session_state.qa_pairs)} Q&A pairs")
            
            with st.expander("📖 Preview questions"):
                for qa in st.session_state.qa_pairs[:5]:
                    st.markdown(f"**Q{qa.get('id')}:** {qa.get('question')}")
                    st.caption(f"Difficulty: {qa.get('difficulty')} | Type: {qa.get('type')}")
                    st.markdown("---")
    
    with tab3:
        st.header("Run Automated Evaluation")
        
        if not st.session_state.document_loaded:
            st.warning("⚠️ Please load a document first (Tab 1)")
        elif not st.session_state.qa_pairs:
            st.warning("⚠️ Please load a Q&A dataset first (Tab 2)")
        else:
            st.markdown(f"**Ready to evaluate:**")
            st.markdown(f"- Document: {len(st.session_state.rag_system.chunks)} chunks loaded")
            st.markdown(f"- Questions: {len(st.session_state.qa_pairs)} Q&A pairs")
            
            num_to_eval = st.slider(
                "Number of questions to evaluate",
                1,
                len(st.session_state.qa_pairs),
                min(10, len(st.session_state.qa_pairs))
            )
            
            if st.button("🚀 Start Evaluation", type="primary"):
                if not api_key:
                    st.error("⚠️ Please provide API key")
                else:
                    evaluator = RAGEvaluator(api_key)
                    
                    st.markdown("### 🔄 Evaluation in Progress...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run evaluation
                    eval_results = evaluator.batch_evaluate(
                        st.session_state.rag_system,
                        st.session_state.qa_pairs[:num_to_eval],
                        progress_bar,
                        status_text
                    )
                    
                    st.session_state.eval_results = eval_results
                    
                    st.success("✅ Evaluation complete!")
                    st.balloons()
    
    with tab4:
        st.header("Evaluation Results")
        
        if st.session_state.eval_results is None:
            st.info("👈 Run evaluation first to see results")
        else:
            results = st.session_state.eval_results
            
            # Overall metrics
            st.subheader("📊 Overall Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = results.get('avg_faithfulness', 0)
                delta = "good" if score >= 4 else "poor"
                st.metric("Faithfulness", f"{score:.2f}/5", delta=delta)
            
            with col2:
                score = results.get('avg_answer_relevancy', 0)
                delta = "good" if score >= 4 else "poor"
                st.metric("Answer Relevancy", f"{score:.2f}/5", delta=delta)
            
            with col3:
                score = results.get('avg_context_relevancy', 0)
                delta = "good" if score >= 4 else "poor"
                st.metric("Context Relevancy", f"{score:.2f}/5", delta=delta)
            
            with col4:
                score = results.get('avg_correctness', 0)
                delta = "good" if score >= 4 else "poor"
                st.metric("Correctness", f"{score:.2f}/5", delta=delta)
            
            # Overall score
            avg_overall = (
                results.get('avg_faithfulness', 0) +
                results.get('avg_answer_relevancy', 0) +
                results.get('avg_context_relevancy', 0) +
                results.get('avg_correctness', 0)
            ) / 4
            
            st.markdown("---")
            st.markdown(f"### 🎯 Overall Score: **{avg_overall:.2f}/5**")
            
            if avg_overall >= 4.0:
                st.success("✅ Excellent performance! Ready for production.")
            elif avg_overall >= 3.5:
                st.info("👍 Good performance. Consider minor improvements.")
            elif avg_overall >= 3.0:
                st.warning("⚠️ Acceptable but needs improvement.")
            else:
                st.error("❌ Poor performance. Significant improvements needed.")
            
            # By difficulty
            st.markdown("---")
            st.subheader("📈 Performance by Difficulty")
            
            if 'by_difficulty' in results:
                diff_cols = st.columns(len(results['by_difficulty']))
                for i, (diff, data) in enumerate(results['by_difficulty'].items()):
                    with diff_cols[i]:
                        st.metric(
                            diff.capitalize(),
                            f"{data['avg_correctness']:.2f}/5",
                            delta=f"{data['count']} questions"
                        )
            
            # By question type
            st.markdown("---")
            st.subheader("📋 Performance by Question Type")
            
            if 'by_type' in results:
                for qtype, data in results['by_type'].items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{qtype}**")
                    with col2:
                        st.markdown(f"{data['avg_correctness']:.2f}/5 ({data['count']} questions)")
            
            # Individual results
            st.markdown("---")
            st.subheader("📝 Individual Question Results")
            
            for i, result in enumerate(results.get('individual_results', [])[:10]):
                with st.expander(f"Q{i+1}: {result['question'][:80]}..."):
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown(f"**Ground Truth:** {result['ground_truth'][:200]}...")
                    st.markdown(f"**Generated:** {result['generated_answer'][:200]}...")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.caption(f"Faithfulness: {result['scores']['faithfulness']}/5")
                    with col2:
                        st.caption(f"Relevancy: {result['scores']['answer_relevancy']}/5")
                    with col3:
                        st.caption(f"Context: {result['scores']['context_relevancy']}/5")
                    with col4:
                        st.caption(f"Correct: {result['scores']['correctness']}/5")
                    
                    if result['scores'].get('explanation'):
                        st.info(result['scores']['explanation'])
            
            # Export
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            json_results = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Download Full Results (JSON)",
                json_results,
                "rag_evaluation_results.json",
                "application/json"
            )


if __name__ == "__main__":
    main()
