"""
Claim/Entity Detection Module

Uses NLP/transformer models to extract key claims, statements,
or named entities from input text.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy


@dataclass
class ExtractedClaim:
    """Represents an extracted claim with its metadata."""
    original_text: str
    claim_text: str
    entities: List[Dict[str, str]]
    keywords: List[str]
    claim_type: str  # 'factual', 'opinion', 'prediction'
    confidence: float


class ClaimExtractor:
    """
    Extracts claims and entities from text using NLP models.

    Combines spaCy for NER and dependency parsing with
    transformer models for more sophisticated claim detection.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", use_transformers: bool = True):
        """
        Initialize the claim extractor.

        Args:
            spacy_model: Name of the spaCy model to use
            use_transformers: Whether to use transformer models for enhanced extraction
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model: {spacy_model}")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

        self.use_transformers = use_transformers

        # Initialize transformer-based NER if requested
        if use_transformers:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"Warning: Could not load transformer NER model: {e}")
                self.ner_pipeline = None
        else:
            self.ner_pipeline = None

        # Claim indicator patterns
        self.factual_indicators = [
            r'\b(announced|declared|stated|confirmed|reported|revealed)\b',
            r'\b(will|shall|is going to|plans to)\b',
            r'\b(has|have|had)\s+(been|announced|started|launched)\b',
            r'\b(according to|as per|sources say)\b',
            r'\b(government|ministry|official|authority)\b',
            r'\b(starting|beginning|from)\s+\w+\s+\d{4}\b',  # Date patterns
        ]

        self.opinion_indicators = [
            r'\b(think|believe|feel|seems|appears|might|may|could|possibly)\b',
            r'\b(in my opinion|personally|arguably)\b',
        ]

        self.prediction_indicators = [
            r'\b(will|going to|expected to|likely to|predicted)\b',
            r'\b(future|upcoming|next year|soon)\b',
        ]

    def extract_entities_spacy(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def extract_entities_transformer(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using transformer model."""
        if not self.ner_pipeline:
            return []

        try:
            results = self.ner_pipeline(text)
            entities = []

            for result in results:
                entities.append({
                    "text": result["word"],
                    "label": result["entity_group"],
                    "start": result["start"],
                    "end": result["end"],
                    "score": result["score"]
                })

            return entities
        except Exception as e:
            print(f"Warning: Transformer NER failed: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords and noun phrases."""
        doc = self.nlp(text)
        keywords = []

        # Extract noun chunks (key phrases)
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text.lower())

        # Extract important nouns and proper nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in keywords:
                keywords.append(token.text.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def classify_claim_type(self, text: str) -> str:
        """Classify the type of claim (factual, opinion, prediction)."""
        text_lower = text.lower()

        # Check for opinion indicators
        for pattern in self.opinion_indicators:
            if re.search(pattern, text_lower):
                return "opinion"

        # Check for prediction indicators
        for pattern in self.prediction_indicators:
            if re.search(pattern, text_lower):
                return "prediction"

        # Default to factual
        return "factual"

    def extract_main_claim(self, text: str) -> str:
        """
        Extract the main claim from the text.

        Uses dependency parsing to identify the core assertion.
        """
        doc = self.nlp(text)

        # Find the root verb and its dependents
        main_clauses = []

        for sent in doc.sents:
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break

            if root:
                # Get the subtree of the root (main clause)
                clause_tokens = list(root.subtree)
                clause_text = " ".join([t.text for t in sorted(clause_tokens, key=lambda x: x.i)])
                main_clauses.append(clause_text)

        # Return the longest clause as the main claim
        if main_clauses:
            return max(main_clauses, key=len)

        return text

    def calculate_confidence(self, text: str, entities: List[Dict[str, str]]) -> float:
        """
        Calculate confidence score for the extracted claim.

        Higher confidence for:
        - More specific entities (dates, organizations, numbers)
        - Factual language patterns
        - Longer, more detailed claims
        """
        score = 0.5  # Base score

        # Entity specificity bonus
        high_value_entities = ["DATE", "ORG", "GPE", "MONEY", "PERCENT", "TIME"]
        for entity in entities:
            if entity.get("label") in high_value_entities:
                score += 0.1

        # Factual indicator bonus
        text_lower = text.lower()
        for pattern in self.factual_indicators:
            if re.search(pattern, text_lower):
                score += 0.05

        # Length bonus (more detailed claims are often more verifiable)
        word_count = len(text.split())
        if word_count > 10:
            score += 0.1
        if word_count > 20:
            score += 0.1

        # Cap at 1.0
        return min(score, 1.0)

    def extract(self, text: str) -> ExtractedClaim:
        """
        Main extraction method.

        Args:
            text: Input text to analyze

        Returns:
            ExtractedClaim object with all extracted information
        """
        # Clean input text
        text = text.strip()

        # Extract entities from both methods
        spacy_entities = self.extract_entities_spacy(text)
        transformer_entities = self.extract_entities_transformer(text) if self.use_transformers else []

        # Merge entities (prefer transformer results for overlapping entities)
        all_entities = self._merge_entities(spacy_entities, transformer_entities)

        # Extract keywords
        keywords = self.extract_keywords(text)

        # Extract main claim
        main_claim = self.extract_main_claim(text)

        # Classify claim type
        claim_type = self.classify_claim_type(text)

        # Calculate confidence
        confidence = self.calculate_confidence(text, all_entities)

        return ExtractedClaim(
            original_text=text,
            claim_text=main_claim,
            entities=all_entities,
            keywords=keywords,
            claim_type=claim_type,
            confidence=confidence
        )

    def _merge_entities(
        self,
        spacy_entities: List[Dict[str, str]],
        transformer_entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Merge entities from different sources, avoiding duplicates."""
        merged = []
        seen_spans = set()

        # Add transformer entities first (usually more accurate)
        for ent in transformer_entities:
            span = (ent.get("start", 0), ent.get("end", 0))
            if span not in seen_spans:
                seen_spans.add(span)
                merged.append(ent)

        # Add spaCy entities that don't overlap
        for ent in spacy_entities:
            span = (ent.get("start", 0), ent.get("end", 0))
            # Check for overlap
            overlaps = False
            for seen_start, seen_end in seen_spans:
                if not (span[1] <= seen_start or span[0] >= seen_end):
                    overlaps = True
                    break

            if not overlaps:
                seen_spans.add(span)
                merged.append(ent)

        return merged

    def extract_multiple_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract multiple claims from a longer text.

        Splits text into sentences and extracts claims from each.
        """
        doc = self.nlp(text)
        claims = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) > 10:  # Skip very short sentences
                claim = self.extract(sent_text)
                claims.append(claim)

        return claims


# Example usage and testing
if __name__ == "__main__":
    extractor = ClaimExtractor(use_transformers=False)  # Set to True if you have transformers installed

    # Test with sample input
    test_text = "The Indian government has announced free electricity to all farmers starting July 2025."

    result = extractor.extract(test_text)

    print("=" * 50)
    print("CLAIM EXTRACTION RESULTS")
    print("=" * 50)
    print(f"Original Text: {result.original_text}")
    print(f"Main Claim: {result.claim_text}")
    print(f"Claim Type: {result.claim_type}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nEntities:")
    for entity in result.entities:
        print(f"  - {entity['text']} ({entity['label']})")
    print(f"\nKeywords: {result.keywords}")
