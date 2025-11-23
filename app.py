"""
Streamlit Web Application for Fact Checker

Provides a user-friendly interface for the AI-Powered Fact Checker.
"""

import os
import sys
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get defaults from environment variables
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
USE_TRANSFORMERS_NER = os.getenv("USE_TRANSFORMERS_NER", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fact_checker import FactChecker, create_fact_checker
from src.claim_extractor import ClaimExtractor


# Page configuration
st.set_page_config(
    page_title="AI Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .verdict-true {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .verdict-false {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .verdict-unverifiable {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .evidence-box {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fact_checker(comparator_type: str = "rule_based", openai_key: str = None):
    """Load and cache the fact checker."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "verified_facts.csv")
    vector_store_path = os.path.join(script_dir, "data", "vector_store")

    # Set API key if provided
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    kwargs = {}
    if comparator_type == "openai":
        kwargs["model"] = OPENAI_MODEL

    checker = FactChecker(
        data_path=data_path,
        vector_store_path=vector_store_path,
        comparator_type=comparator_type,
        use_transformers_ner=USE_TRANSFORMERS_NER,
        **kwargs
    )

    checker.index_facts()
    return checker


def get_verdict_color(verdict: str) -> str:
    """Get the CSS class for the verdict."""
    if verdict == "True":
        return "verdict-true"
    elif verdict == "False":
        return "verdict-false"
    else:
        return "verdict-unverifiable"


def get_verdict_emoji(verdict: str) -> str:
    """Get emoji for verdict."""
    emoji_map = {
        "True": "✅",
        "False": "❌",
        "Unverifiable": "🤷"
    }
    return emoji_map.get(verdict, "🤷")


def main():
    # Header
    st.title("🔍 AI-Powered Fact Checker")
    st.markdown("*Verify claims against a database of trusted facts using AI*")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Comparator selection
    model_options = {"OpenAI": "openai", "Ollama": "ollama"}
    selected_model = st.sidebar.selectbox(
        "Select AI Model",
        list(model_options.keys()),
        help="Choose the comparison method"
    )
    comparator_type = model_options[selected_model]

    # API key input for OpenAI
    openai_key = None
    if comparator_type == "openai":
        # Check if API key is already in environment
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            openai_key = env_key
            st.sidebar.success("API key loaded from .env file")
        else:
            openai_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key (or add it to .env file)"
            )
            if not openai_key:
                st.sidebar.warning("Please enter your OpenAI API key or add it to .env file")

    # Top-K results slider
    top_k = st.sidebar.slider(
        "Number of facts to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="How many similar facts to retrieve for comparison"
    )

    # Similarity threshold
    similarity_threshold = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum similarity score to include facts (filters out low-relevance facts)"
    )

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score for a verdict to be trusted (below this marks as Unverifiable)"
    )

    # Load fact checker
    try:
        with st.spinner("Loading fact checker..."):
            checker = load_fact_checker(
                comparator_type=comparator_type if comparator_type != "openai" or openai_key else "rule_based",
                openai_key=openai_key
            )
            checker.top_k = top_k
    except Exception as e:
        st.error(f"Error loading fact checker: {e}")
        st.stop()

    # Display stats in sidebar
    st.sidebar.header("📊 Database Stats")
    stats = checker.get_stats()
    st.sidebar.metric("Total Facts", stats["total_facts"])

    # Main content area
    st.header("📝 Enter a Claim to Verify")

    # Sample claims
    st.markdown("**Sample claims to try:**")
    sample_claims = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "India's Chandrayaan-3 successfully landed on the Moon in August 2023.",
        "PM-KISAN provides Rs 10000 per year to farmers.",
        "The GST was implemented in India in 2018.",
    ]

    for i, claim in enumerate(sample_claims):
        if st.button(f"📌 {claim}", key=f"sample_{i}"):
            st.session_state.input_text = claim

    # Text input
    input_text = st.text_area(
        "Enter news post or social media statement:",
        value=st.session_state.get("input_text", ""),
        height=100,
        placeholder="e.g., The Indian government has announced free electricity to all farmers starting July 2025."
    )

    # Check button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button("🔎 Verify Claim", type="primary", use_container_width=True)

    # Process and display results
    if check_button and input_text:
        with st.spinner("Analyzing claim..."):
            try:
                result = checker.check(input_text, score_threshold=similarity_threshold, confidence_threshold=confidence_threshold)

                # Display results
                st.header("📋 Verification Results")

                # Verdict banner
                verdict_class = get_verdict_color(result.verdict)
                verdict_emoji = get_verdict_emoji(result.verdict)

                st.markdown(f"""
                <div class="{verdict_class}">
                    <h2 style="margin: 0;">{verdict_emoji} {result.verdict}</h2>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em;">Confidence: {result.confidence:.0%}</p>
                </div>
                """, unsafe_allow_html=True)

                # Tabs for detailed results
                tab1, tab2, tab3, tab4 = st.tabs(["💡 Reasoning", "📚 Evidence", "🔍 Analysis", "📄 JSON Output"])

                with tab1:
                    st.subheader("Claim")
                    st.info(result.extracted_claim)

                    st.subheader("Reasoning")
                    st.write(result.reasoning)

                with tab2:
                    st.subheader("Supporting Evidence")
                    if result.evidence:
                        for i, ev in enumerate(result.evidence, 1):
                            st.markdown(f"""
                            <div class="evidence-box">
                                <strong>Evidence {i}:</strong><br>
                                {ev}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No direct evidence found in the database.")

                    st.subheader("Retrieved Facts (by similarity)")
                    for i, fact in enumerate(result.retrieved_facts, 1):
                        score = fact.get("score", 0)
                        source = fact.get("source", "Unknown")

                        # Color code by score
                        if score >= 0.7:
                            color = "🟢"
                        elif score >= 0.5:
                            color = "🟡"
                        else:
                            color = "🔴"

                        with st.expander(f"{color} Fact {i} - Score: {score:.3f}"):
                            st.write(f"**Fact:** {fact.get('fact', '')}")
                            st.write(f"**Source:** {source}")
                            st.write(f"**Similarity Score:** {score:.4f}")

                with tab3:
                    st.subheader("Named Entities")
                    if result.entities:
                        for entity in result.entities:
                            st.write(f"• **{entity.get('text', '')}** ({entity.get('label', '')})")
                    else:
                        st.write("No entities extracted")

                with tab4:
                    st.subheader("Output")
                    formatted = result.format_output()
                    st.code(json.dumps(formatted, indent=2), language="json")

                    st.subheader("Full Result Object")
                    st.code(result.to_json(), language="json")

                # Feedback section (bonus feature)
                st.header("📣 Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful", use_container_width=True):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful", use_container_width=True):
                        st.info("We'll work on improving!")
                with col3:
                    if st.button("🚩 Report Issue", use_container_width=True):
                        st.warning("Thank you for reporting. We'll review this result.")

            except Exception as e:
                st.error(f"Error checking claim: {e}")
                st.exception(e)

    elif check_button:
        st.warning("Please enter a claim to verify.")



if __name__ == "__main__":
    main()
