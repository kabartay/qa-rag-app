import streamlit as st
import anthropic
import json
from typing import List, Dict, Tuple
import time
from datetime import datetime

# RAG System Evaluator
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
    
    def batch_evaluate(self, rag_system, qa_pairs: List[Dict]) -> Dict:
        """Evaluate RAG system on multiple Q&A pairs"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, qa in enumerate(qa_pairs):
            status_text.text(f"Evaluating {i+1}/{len(qa_pairs)}...")
            
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
        
        status_text.text("Evaluation complete!")
        
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
            aggregate['by_difficulty'][diff] = {
                'count': len(diff_results),
                'avg_correctness': sum(r['scores']['correctness'] for r in diff_results) / len(diff_results)
            }
        
        # By question type
        types = set(r['type'] for r in results)
        aggregate['by_type'] = {}
        
        for qtype in types:
            type_results = [r for r in results if r['type'] == qtype]
            aggregate['by_type'][qtype] = {
                'count': len(type_results),
                'avg_correctness': sum(r['scores']['correctness'] for r in type_results) / len(type_results)
            }
        
        return aggregate


def main():
    st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")
    
    st.title("📊 RAG System Evaluator")
    st.markdown("Comprehensive evaluation of RAG systems using groundtruth datasets")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Anthropic API Key", type="password")
        
        st.markdown("---")
        st.header("Evaluation Metrics")
        st.markdown("**Faithfulness**: Answer supported by context")
        st.markdown("**Answer Relevancy**: Answer addresses question")
        st.markdown("**Context Relevancy**: Retrieved context is relevant")
        st.markdown("**Correctness**: Compared to ground truth")
        
        st.markdown("---")
        st.markdown("### Evaluation Methods")
        st.markdown("1. **LLM-as-Judge**: Claude evaluates quality")
        st.markdown("2. **Multi-dimensional**: 4 key metrics")
        st.markdown("3. **Stratified**: By difficulty & question type")
    
    # Tabs for different evaluation modes
    tab1, tab2, tab3 = st.tabs(["📁 Load Data", "🧪 Run Evaluation", "📈 Results"])
    
    with tab1:
        st.header("Load Groundtruth Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Upload Q&A JSON file", type=['json'])
            
            if uploaded_file:
                try:
                    qa_data = json.load(uploaded_file)
                    st.session_state.qa_pairs = qa_data
                    st.success(f"✅ Loaded {len(qa_data)} Q&A pairs")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        with col2:
            st.markdown("**Sample Data Format:**")
            st.code('''[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "difficulty": "easy",
    "type": "factual"
  }
]''', language='json')
        
        if 'qa_pairs' in st.session_state and st.session_state.qa_pairs:
            st.markdown(f"**Dataset loaded:** {len(st.session_state.qa_pairs)} Q&A pairs")
            
            # Preview
            with st.expander("Preview data"):
                st.json(st.session_state.qa_pairs[:3])
    
    with tab2:
        st.header("Run Evaluation")
        
        if 'qa_pairs' not in st.session_state or not st.session_state.qa_pairs:
            st.warning("⚠️ Please load a groundtruth dataset first")
        else:
            st.markdown(f"**Ready to evaluate:** {len(st.session_state.qa_pairs)} questions")
            
            # RAG system selection
            st.markdown("### Select RAG System")
            st.info("🔧 In production, you would connect to your RAG system API here. For this demo, evaluation logic is shown.")
            
            num_to_eval = st.slider(
                "Number of questions to evaluate",
                1,
                min(len(st.session_state.qa_pairs), 20),
                min(10, len(st.session_state.qa_pairs))
            )
            
            if st.button("🚀 Start Evaluation", type="primary", disabled=not api_key):
                if not api_key:
                    st.error("Please provide API key")
                else:
                    evaluator = RAGEvaluator(api_key)
                    
                    st.markdown("### Evaluation in Progress...")
                    st.info("This would evaluate your RAG system. For demo, showing evaluation framework.")
                    
                    # Mock evaluation for demo
                    st.session_state.eval_results = {
                        'total_questions': num_to_eval,
                        'avg_faithfulness': 4.2,
                        'avg_answer_relevancy': 4.5,
                        'avg_context_relevancy': 3.8,
                        'avg_correctness': 4.0,
                        'by_difficulty': {
                            'easy': {'count': 4, 'avg_correctness': 4.5},
                            'medium': {'count': 4, 'avg_correctness': 4.0},
                            'hard': {'count': 2, 'avg_correctness': 3.5}
                        }
                    }
                    
                    st.success("✅ Evaluation complete!")
                    st.balloons()
    
    with tab3:
        st.header("Evaluation Results")
        
        if 'eval_results' not in st.session_state:
            st.info("👈 Run evaluation first to see results")
        else:
            results = st.session_state.eval_results
            
            # Overall metrics
            st.subheader("📊 Overall Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = results.get('avg_faithfulness', 0)
                st.metric("Faithfulness", f"{score:.2f}/5", delta=f"{score-3:.1f}")
            
            with col2:
                score = results.get('avg_answer_relevancy', 0)
                st.metric("Answer Relevancy", f"{score:.2f}/5", delta=f"{score-3:.1f}")
            
            with col3:
                score = results.get('avg_context_relevancy', 0)
                st.metric("Context Relevancy", f"{score:.2f}/5", delta=f"{score-3:.1f}")
            
            with col4:
                score = results.get('avg_correctness', 0)
                st.metric("Correctness", f"{score:.2f}/5", delta=f"{score-3:.1f}")
            
            # By difficulty
            st.subheader("📈 Performance by Difficulty")
            
            if 'by_difficulty' in results:
                cols = st.columns(len(results['by_difficulty']))
                for i, (diff, data) in enumerate(results['by_difficulty'].items()):
                    with cols[i]:
                        st.metric(
                            diff.capitalize(),
                            f"{data['avg_correctness']:.2f}/5",
                            delta=f"{data['count']} questions"
                        )
            
            # Export
            st.subheader("💾 Export Results")
            
            json_results = json.dumps(results, indent=2)
            st.download_button(
                "📥 Download Results (JSON)",
                json_results,
                "rag_evaluation_results.json",
                "application/json"
            )


if __name__ == "__main__":
    main()