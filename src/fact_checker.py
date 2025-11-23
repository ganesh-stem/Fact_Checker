"""
Main Fact Checker Pipeline

Combines all components (claim extraction, embedding, retrieval, LLM comparison)
into a unified fact-checking pipeline.
"""

import os
import json
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime

# Get confidence threshold from environment variable
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

from .claim_extractor import ClaimExtractor, ExtractedClaim
from .fact_manager import FactBaseManager
from .embeddings import RetrievalSystem, RetrievalResult
from .llm_comparator import (
    get_comparator,
    LLMComparator,
    VerificationResult,
    RuleBasedComparator
)


@dataclass
class FactCheckResult:
    """Complete result of fact-checking a statement."""
    input_text: str
    extracted_claim: str
    entities: List[Dict]
    verdict: str
    confidence: float
    evidence: List[str]
    reasoning: str
    retrieved_facts: List[Dict]
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    def get_verdict_emoji(self) -> str:
        """Get emoji for the verdict."""
        emoji_map = {
            "True": "✅",
            "False": "❌",
            "Unverifiable": "🤷"
        }
        return emoji_map.get(self.verdict, "🤷")

    def format_output(self) -> Dict:
        """Format output as specified in the task requirements."""
        return {
            "verdict": self.verdict,
            "evidence": self.evidence,
            "reasoning": self.reasoning
        }


class FactChecker:
    """
    Main fact-checking pipeline.

    Orchestrates:
    1. Claim extraction from input text
    2. Embedding and retrieval of relevant facts
    3. LLM-based comparison and verdict generation
    """

    def __init__(
        self,
        data_path: str = "data/verified_facts.csv",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "data/vector_store",
        comparator_type: Literal["openai", "ollama", "rule_based"] = "rule_based",
        use_transformers_ner: bool = False,
        top_k: int = 5,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        **comparator_kwargs
    ):
        """
        Initialize the fact checker.

        Args:
            data_path: Path to the verified facts CSV
            embedding_model: Name of the embedding model
            vector_store_path: Path for vector store persistence
            comparator_type: Type of LLM comparator to use
            use_transformers_ner: Whether to use transformer-based NER
            top_k: Number of facts to retrieve
            confidence_threshold: Minimum confidence for a verdict (default from CONFIDENCE_THRESHOLD env var)
            **comparator_kwargs: Additional arguments for the comparator
        """
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        # Initialize components
        print("Initializing Fact Checker...")

        print("  - Loading claim extractor...")
        self.claim_extractor = ClaimExtractor(use_transformers=use_transformers_ner)

        print("  - Loading fact base...")
        self.fact_manager = FactBaseManager(data_path)
        self.fact_manager.load_facts()

        print("  - Initializing retrieval system...")
        self.retrieval_system = RetrievalSystem(
            embedding_model=embedding_model,
            vector_store_type="faiss",
            persist_path=vector_store_path
        )

        print("  - Setting up comparator...")
        self.comparator = get_comparator(comparator_type, **comparator_kwargs)

        # Track initialization state
        self._indexed = False

        print("Fact Checker initialized!")

    def index_facts(self, force_reindex: bool = False):
        """
        Index all facts from the fact base.

        Args:
            force_reindex: Whether to force re-indexing even if index exists
        """
        # Check if index already exists
        if not force_reindex and os.path.exists(f"{self.vector_store_path}.faiss"):
            print("Loading existing vector index...")
            self.retrieval_system.load(self.vector_store_path)
            self._indexed = True
            return

        # Get facts with metadata
        facts_data = self.fact_manager.get_facts_with_metadata()

        if not facts_data:
            print("Warning: No facts found in database!")
            return

        # Prepare for indexing
        facts = [f["fact"] for f in facts_data]
        metadata = [
            {
                "id": f.get("id"),
                "source": f.get("source", "Unknown")
            }
            for f in facts_data
        ]

        # Index
        self.retrieval_system.index_facts(facts, metadata)

        # Save index
        self.retrieval_system.save(self.vector_store_path)
        self._indexed = True

    def check(self, text: str, score_threshold: float = 0.0, confidence_threshold: float = None) -> FactCheckResult:
        """
        Check a statement for factual accuracy.

        Args:
            text: Input text (news post, social media statement, etc.)
            score_threshold: Minimum similarity score to include a fact (0.0-1.0)
            confidence_threshold: Minimum confidence for verdict (defaults to self.confidence_threshold)

        Returns:
            FactCheckResult with verdict and reasoning
        """
        # Use instance threshold if not provided
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        # Ensure facts are indexed
        if not self._indexed:
            self.index_facts()

        # Step 1: Extract claim
        extracted = self.claim_extractor.extract(text)

        # Step 2: Retrieve relevant facts
        results = self.retrieval_system.retrieve(extracted.claim_text, self.top_k)

        # Convert results to format expected by comparator
        retrieved_facts = [
            {
                "fact": r.fact,
                "score": r.score,
                "source": r.metadata.get("source", "Unknown"),
                "metadata": r.metadata
            }
            for r in results
        ]

        # Filter facts by score threshold
        filtered_facts = [f for f in retrieved_facts if f["score"] >= score_threshold]

        # If no facts pass threshold, mark as unverifiable
        if not filtered_facts:
            return FactCheckResult(
                input_text=text,
                extracted_claim=extracted.claim_text,
                entities=[e for e in extracted.entities],
                verdict="Unverifiable",
                confidence=0.0,
                evidence=[],
                reasoning=f"No facts found with similarity score above {score_threshold:.0%} threshold.",
                retrieved_facts=retrieved_facts,
                timestamp=datetime.now().isoformat()
            )

        # Step 3: Compare and generate verdict (using filtered facts)
        verification = self.comparator.compare(extracted.claim_text, filtered_facts)

        # Step 4: Apply confidence threshold
        verdict = verification.verdict
        reasoning = verification.reasoning
        if verification.confidence < confidence_threshold and verdict != "Unverifiable":
            reasoning = f"Confidence ({verification.confidence:.0%}) below threshold ({confidence_threshold:.0%}). {reasoning}"
            verdict = "Unverifiable"

        # Step 5: Build result (use filtered_facts, not all retrieved_facts)
        result = FactCheckResult(
            input_text=text,
            extracted_claim=extracted.claim_text,
            entities=[e for e in extracted.entities],
            verdict=verdict,
            confidence=verification.confidence,
            evidence=verification.evidence,
            reasoning=reasoning,
            retrieved_facts=filtered_facts,
            timestamp=datetime.now().isoformat()
        )

        return result

    def check_batch(self, texts: List[str]) -> List[FactCheckResult]:
        """
        Check multiple statements.

        Args:
            texts: List of input texts

        Returns:
            List of FactCheckResult objects
        """
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Checking {i}/{len(texts)}...")
            result = self.check(text)
            results.append(result)
        return results

    def get_retrieval_only(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Get only the retrieval results without LLM comparison.

        Useful for debugging or when LLM is not available.

        Args:
            query: Query string
            top_k: Number of results (defaults to self.top_k)

        Returns:
            List of RetrievalResult objects
        """
        if not self._indexed:
            self.index_facts()

        return self.retrieval_system.retrieve(query, top_k or self.top_k)

    def add_fact(self, fact: str, source: str, **kwargs):
        """
        Add a new fact to the database and re-index.

        Args:
            fact: The fact text
            source: Source of the fact
            **kwargs: Additional metadata (date, category, keywords)
        """
        self.fact_manager.add_fact(fact, source, **kwargs)
        self.fact_manager.save_facts()

        # Re-index with new fact
        self.index_facts(force_reindex=True)

    def get_stats(self) -> Dict:
        """Get statistics about the fact checker."""
        return {
            "total_facts": len(self.fact_manager.facts_df) if self.fact_manager.facts_df is not None else 0,
            "indexed": self._indexed,
            "comparator_type": type(self.comparator).__name__
        }


def create_fact_checker(
    use_openai: bool = False,
    use_ollama: bool = False,
    openai_model: str = "gpt-3.5-turbo",
    ollama_model: str = "mistral",
    data_path: str = None
) -> FactChecker:
    """
    Factory function to create a configured FactChecker.

    Args:
        use_openai: Use OpenAI for comparison
        use_ollama: Use Ollama for comparison
        openai_model: OpenAI model name
        ollama_model: Ollama model name
        data_path: Path to facts CSV

    Returns:
        Configured FactChecker instance
    """
    # Determine comparator type
    if use_openai and os.getenv("OPENAI_API_KEY"):
        comparator_type = "openai"
        kwargs = {"model": openai_model}
    elif use_ollama:
        comparator_type = "ollama"
        kwargs = {"model": ollama_model}
    else:
        comparator_type = "rule_based"
        kwargs = {}

    # Determine data path
    if data_path is None:
        # Try to find data file relative to this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(module_dir, "..", "data", "verified_facts.csv")

    return FactChecker(
        data_path=data_path,
        comparator_type=comparator_type,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Determine path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "verified_facts.csv")
    vector_store_path = os.path.join(script_dir, "..", "data", "vector_store")

    print("=" * 60)
    print("AI-Powered Fact Checker Demo")
    print("=" * 60)

    # Initialize fact checker
    checker = FactChecker(
        data_path=data_path,
        vector_store_path=vector_store_path,
        comparator_type="rule_based"  # Use rule_based for demo without API keys
    )

    # Index facts
    checker.index_facts()

    # Test claims
    test_claims = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "India's Chandrayaan-3 successfully landed on the Moon in August 2023.",
        "The GST was implemented in India on July 1, 2017.",
        "PM-KISAN provides Rs 10000 per year to farmers.",  # Incorrect amount
    ]

    print("\n" + "=" * 60)
    print("FACT-CHECKING RESULTS")
    print("=" * 60)

    for claim in test_claims:
        print(f"\n{'─' * 60}")
        print(f"INPUT: {claim}")
        print("─" * 60)

        result = checker.check(claim)

        print(f"\n{result.get_verdict_emoji()} VERDICT: {result.verdict}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nReasoning: {result.reasoning}")

        if result.evidence:
            print(f"\nEvidence:")
            for ev in result.evidence[:3]:
                print(f"  • {ev[:100]}...")

        print(f"\nFormatted Output:")
        print(json.dumps(result.format_output(), indent=2))
