"""
Main CLI Entry Point for Fact Checker

Provides command-line interface for the AI-Powered Fact Checker.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get defaults from environment variables
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
DEFAULT_MODEL = "ollama" if USE_LOCAL_LLM else "openai"
DEFAULT_TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
USE_TRANSFORMERS_NER = os.getenv("USE_TRANSFORMERS_NER", "false").lower() == "true"

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fact_checker import FactChecker


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Fact Checker - Verify claims against trusted facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "The Indian government announced free electricity to farmers"
  python main.py --file claims.txt
  python main.py --interactive
  python main.py --model openai "Your claim here"
        """
    )

    parser.add_argument(
        "claim",
        nargs="?",
        help="The claim to verify"
    )

    parser.add_argument(
        "--file", "-f",
        help="File containing claims (one per line)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--model", "-m",
        choices=["openai", "ollama"],
        default=DEFAULT_MODEL,
        help=f"Model/comparator type to use (default: {DEFAULT_MODEL}, set via USE_LOCAL_LLM in .env)"
    )

    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of facts to retrieve (default: {DEFAULT_TOP_K}, set via TOP_K_RESULTS in .env)"
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with all details"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "verified_facts.csv")
    vector_store_path = os.path.join(script_dir, "data", "vector_store")

    # Initialize fact checker
    print("Initializing Fact Checker...")
    checker = FactChecker(
        data_path=data_path,
        vector_store_path=vector_store_path,
        comparator_type=args.model,
        top_k=args.top_k,
        use_transformers_ner=USE_TRANSFORMERS_NER
    )
    checker.index_facts()

    def check_and_display(claim: str):
        """Check a claim and display results."""
        result = checker.check(claim)

        if args.json:
            print(result.to_json())
        else:
            emoji = result.get_verdict_emoji()
            print("\n" + "=" * 60)
            print(f"CLAIM: {claim}")
            print("=" * 60)
            print(f"\n{emoji} VERDICT: {result.verdict}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"\nReasoning: {result.reasoning}")

            if result.evidence:
                print("\nEvidence:")
                for i, ev in enumerate(result.evidence, 1):
                    print(f"  {i}. {ev}")

            if args.verbose:
                print(f"Entities: {[e.get('text') for e in result.entities]}")

                print("\nRetrieved Facts:")
                for i, fact in enumerate(result.retrieved_facts, 1):
                    print(f"  [{i}] Score: {fact.get('score', 0):.3f}")
                    print(f"      {fact.get('fact', '')[:80]}...")

            print("\n" + "-" * 60)
            print("Formatted Output:")
            print(json.dumps(result.format_output(), indent=2))

    # Handle different modes
    if args.interactive:
        print("\n" + "=" * 60)
        print("AI-Powered Fact Checker - Interactive Mode")
        print("=" * 60)
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                claim = input("\nEnter claim to verify: ").strip()
                if claim.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if claim:
                    check_and_display(claim)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            claims = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(claims)} claims from {args.file}")
        for claim in claims:
            check_and_display(claim)

    elif args.claim:
        check_and_display(args.claim)

    else:
        # Demo mode with sample claims
        print("\n" + "=" * 60)
        print("AI-Powered Fact Checker - Demo Mode")
        print("=" * 60)

        sample_claims = [
            "The Indian government has announced free electricity to all farmers starting July 2025.",
            "India's Chandrayaan-3 successfully landed on the Moon in August 2023.",
        ]

        for claim in sample_claims:
            check_and_display(claim)


if __name__ == "__main__":
    main()
