"""
LLM-Powered Comparison Module

Uses LLMs (OpenAI GPT, local models via Ollama) to compare claims
against retrieved facts and generate verdicts with reasoning.
"""

import os
import json
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


class Verdict(Enum):
    """Possible verdicts for claim verification."""
    TRUE = "True"
    LIKELY_TRUE = "Likely True"
    UNVERIFIABLE = "Unverifiable"
    LIKELY_FALSE = "Likely False"
    FALSE = "False"


@dataclass
class VerificationResult:
    """Result of claim verification."""
    verdict: str
    confidence: float
    evidence: List[str]
    reasoning: str
    claim: str
    retrieved_facts: List[Dict]


class LLMComparator:
    """
    Base class for LLM-based fact comparison.

    Constructs prompts and processes LLM responses.
    """

    SYSTEM_PROMPT = """You are a fact-checker. Compare the claim against the retrieved facts and respond with ONLY a JSON object.

Response format:
{"verdict": "True/False/Unverifiable", "confidence": 0.0-1.0, "evidence": ["copy the full text of relevant facts here"], "reasoning": "brief explanation"}

IMPORTANT: In the "evidence" array, copy the FULL TEXT of the relevant facts, not just numbers.

Verdicts (use ONLY these 3):
- True: Claim is supported by the facts (includes cases where claim is less specific but consistent with facts)
- False: Claim is CONTRADICTED by the facts (dates/numbers/details are WRONG, not just less specific)
- Unverifiable: Not enough info to verify or refute

CRITICAL: A less specific claim is NOT false. Examples:
- Claim: "August 2023" vs Fact: "August 23, 2023" = TRUE (August 23 IS in August 2023)
- Claim: "over 100 startups" vs Fact: "over 100,000 startups" = FALSE (100 vs 100,000 is wrong)
- Claim: "2001" vs Fact: "2021" = FALSE (wrong year)

Confidence guidelines:
- 0.9-1.0: Very high confidence (exact match or clear contradiction)
- 0.7-0.9: High confidence (strong evidence supports verdict)
- 0.5-0.7: Moderate confidence (some evidence but not definitive)
- 0.3-0.5: Low confidence (weak evidence)
- 0.0-0.3: Very low confidence (almost no relevant evidence)"""

    def construct_prompt(self, claim: str, retrieved_facts: List[Dict]) -> str:
        """
        Construct the comparison prompt.

        Args:
            claim: The claim to verify
            retrieved_facts: List of retrieved facts with scores

        Returns:
            Formatted prompt string
        """
        facts_text = ""
        for i, fact in enumerate(retrieved_facts, 1):
            score = fact.get("score", 0)
            text = fact.get("fact", fact.get("text", ""))
            source = fact.get("source", "Unknown")
            facts_text += f"{i}. [Score: {score:.3f}] {text}\n   Source: {source}\n\n"

        prompt = f"""CLAIM TO VERIFY:
"{claim}"

RETRIEVED FACTS FROM TRUSTED SOURCES:
{facts_text}

Based on the above facts, verify the claim. Respond with a JSON object containing:
- verdict: "True", "False", or "Unverifiable" (use ONLY these 3 values)
- confidence: number between 0 and 1 (use 0.8+ for clear matches or contradictions)
- evidence: list of relevant fact texts that support your verdict
- reasoning: explanation of your verdict

JSON Response:"""

        return prompt

    def parse_response(self, response: str, claim: str, retrieved_facts: List[Dict]) -> VerificationResult:
        """
        Parse LLM response into VerificationResult.

        Args:
            response: Raw LLM response
            claim: Original claim
            retrieved_facts: Retrieved facts

        Returns:
            VerificationResult object
        """
        # Try to extract JSON from response
        try:
            # Handle cases where response has markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                else:
                    json_str = response

            result = json.loads(json_str)

            return VerificationResult(
                verdict=result.get("verdict", "Unverifiable"),
                confidence=float(result.get("confidence", 0.5)),
                evidence=result.get("evidence", []),
                reasoning=result.get("reasoning", "Unable to parse reasoning"),
                claim=claim,
                retrieved_facts=retrieved_facts
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for unparseable responses
            return VerificationResult(
                verdict="Unverifiable",
                confidence=0.3,
                evidence=[],
                reasoning=f"Failed to parse LLM response: {str(e)}. Raw response: {response[:200]}",
                claim=claim,
                retrieved_facts=retrieved_facts
            )


class OpenAIComparator(LLMComparator):
    """
    LLM comparator using OpenAI GPT models.
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize OpenAI comparator.

        Args:
            model: OpenAI model name (defaults to OPENAI_MODEL env var or gpt-4o-mini)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if OpenAI is None:
            raise ImportError("openai not installed. Run: pip install openai")

        self.model = model
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def compare(self, claim: str, retrieved_facts: List[Dict]) -> VerificationResult:
        """
        Compare claim against facts using OpenAI.

        Args:
            claim: Claim to verify
            retrieved_facts: Retrieved facts

        Returns:
            VerificationResult
        """
        prompt = self.construct_prompt(claim, retrieved_facts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = response.choices[0].message.content
            return self.parse_response(response_text, claim, retrieved_facts)

        except Exception as e:
            return VerificationResult(
                verdict="Unverifiable",
                confidence=0.0,
                evidence=[],
                reasoning=f"OpenAI API error: {str(e)}",
                claim=claim,
                retrieved_facts=retrieved_facts
            )


class OllamaComparator(LLMComparator):
    """
    LLM comparator using local models via Ollama.

    Supports models like Mistral, Llama2, etc.
    """

    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama comparator.

        Args:
            model: Ollama model name (defaults to LOCAL_LLM_MODEL env var)
            base_url: Ollama server URL
        """
        if requests is None:
            raise ImportError("requests not installed. Run: pip install requests")

        self.model = model or os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
        self.base_url = base_url
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

    def compare(self, claim: str, retrieved_facts: List[Dict]) -> VerificationResult:
        """
        Compare claim against facts using Ollama.

        Args:
            claim: Claim to verify
            retrieved_facts: Retrieved facts

        Returns:
            VerificationResult
        """
        prompt = self.construct_prompt(claim, retrieved_facts)
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"

        try:
            # Use the generate API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 8192
                    }
                },
                timeout=300  # Increased timeout for slower models
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                return self.parse_response(response_text, claim, retrieved_facts)
            else:
                # Try to get error details
                error_detail = ""
                try:
                    error_detail = response.json().get("error", response.text[:200])
                except:
                    error_detail = response.text[:200]
                raise Exception(f"Ollama returned status {response.status_code}: {error_detail}")

        except requests.exceptions.ConnectionError:
            return VerificationResult(
                verdict="Unverifiable",
                confidence=0.0,
                evidence=[],
                reasoning="Could not connect to Ollama. Make sure Ollama is running (ollama serve)",
                claim=claim,
                retrieved_facts=retrieved_facts
            )
        except requests.exceptions.Timeout:
            return VerificationResult(
                verdict="Unverifiable",
                confidence=0.0,
                evidence=[],
                reasoning="Ollama request timed out. The model may be loading or the prompt is too long.",
                claim=claim,
                retrieved_facts=retrieved_facts
            )
        except Exception as e:
            return VerificationResult(
                verdict="Unverifiable",
                confidence=0.0,
                evidence=[],
                reasoning=f"Ollama error: {str(e)}",
                claim=claim,
                retrieved_facts=retrieved_facts
            )


class RuleBasedComparator(LLMComparator):
    """
    Simple rule-based comparator for cases without LLM access.

    Uses keyword matching and similarity scores.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize rule-based comparator.

        Args:
            similarity_threshold: Minimum similarity for supporting evidence
        """
        self.similarity_threshold = similarity_threshold

    def compare(self, claim: str, retrieved_facts: List[Dict]) -> VerificationResult:
        """
        Compare claim against facts using rules.

        Args:
            claim: Claim to verify
            retrieved_facts: Retrieved facts

        Returns:
            VerificationResult
        """
        claim_lower = claim.lower()

        # Analyze retrieved facts
        supporting = []
        contradicting = []
        related = []

        for fact in retrieved_facts:
            text = fact.get("fact", fact.get("text", "")).lower()
            score = fact.get("score", 0)

            if score >= self.similarity_threshold:
                # Check for contradiction indicators
                if self._check_contradiction(claim_lower, text):
                    contradicting.append(fact)
                else:
                    supporting.append(fact)
            elif score >= 0.5:
                related.append(fact)

        # Determine verdict
        if contradicting:
            verdict = "Likely False" if supporting else "False"
            confidence = min(0.9, max(f.get("score", 0) for f in contradicting))
            evidence = [f.get("fact", f.get("text", "")) for f in contradicting]
            reasoning = f"Found {len(contradicting)} contradicting fact(s)."

        elif supporting:
            if len(supporting) >= 2:
                verdict = "True"
                confidence = min(0.95, sum(f.get("score", 0) for f in supporting) / len(supporting))
            else:
                verdict = "Likely True"
                confidence = supporting[0].get("score", 0.6)

            evidence = [f.get("fact", f.get("text", "")) for f in supporting]
            reasoning = f"Found {len(supporting)} supporting fact(s) with high similarity."

        elif related:
            verdict = "Unverifiable"
            confidence = 0.4
            evidence = [f.get("fact", f.get("text", "")) for f in related[:2]]
            reasoning = "Found related facts but insufficient evidence to verify."

        else:
            verdict = "Unverifiable"
            confidence = 0.2
            evidence = []
            reasoning = "No relevant facts found in the database."

        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            claim=claim,
            retrieved_facts=retrieved_facts
        )

    def _check_contradiction(self, claim: str, fact: str) -> bool:
        """Check if fact contradicts the claim."""
        contradiction_patterns = [
            ("free", "no free"),
            ("free", "not free"),
            ("all", "not all"),
            ("national", "state"),
            ("announced", "not announced"),
            ("no ", "has "),
        ]

        for positive, negative in contradiction_patterns:
            if positive in claim and negative in fact:
                return True
            if negative in claim and positive in fact:
                return True

        return False


def get_comparator(
    comparator_type: Literal["openai", "ollama", "rule_based"] = "rule_based",
    **kwargs
) -> LLMComparator:
    """
    Factory function to get the appropriate comparator.

    Args:
        comparator_type: Type of comparator to use
        **kwargs: Additional arguments for the comparator

    Returns:
        LLMComparator instance
    """
    if comparator_type == "openai":
        return OpenAIComparator(**kwargs)
    elif comparator_type == "ollama":
        return OllamaComparator(**kwargs)
    elif comparator_type == "rule_based":
        return RuleBasedComparator(**kwargs)
    else:
        raise ValueError(f"Unknown comparator type: {comparator_type}")


# Example usage
if __name__ == "__main__":
    # Test with rule-based comparator
    comparator = RuleBasedComparator()

    claim = "The Indian government has announced free electricity to all farmers starting July 2025."

    retrieved_facts = [
        {
            "fact": "There is no government scheme announced to provide free electricity to all farmers in India as of 2024.",
            "score": 0.85,
            "source": "PIB India"
        },
        {
            "fact": "PM-KUSUM scheme provides subsidized solar panels for farmers, not free electricity.",
            "score": 0.75,
            "source": "MNRE"
        },
        {
            "fact": "State-specific electricity subsidies exist but there is no national free electricity scheme for farmers.",
            "score": 0.70,
            "source": "Ministry of Power"
        }
    ]

    result = comparator.compare(claim, retrieved_facts)

    print("=" * 50)
    print("VERIFICATION RESULT")
    print("=" * 50)
    print(f"Claim: {result.claim}")
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"\nEvidence:")
    for ev in result.evidence:
        print(f"  - {ev}")
