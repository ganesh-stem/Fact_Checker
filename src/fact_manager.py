"""
Fact Base Manager Module

Manages loading and saving verified facts from CSV files.
"""

import os
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd


@dataclass
class FactEntry:
    """Represents a verified fact entry."""
    id: int
    fact: str
    source: str


class FactBaseManager:
    """
    Manages the fact database - loading, saving, and updating.
    """

    def __init__(self, data_path: str = "data/verified_facts.csv"):
        self.data_path = data_path
        self.facts_df = None

    def load_facts(self) -> pd.DataFrame:
        """Load facts from CSV file."""
        if os.path.exists(self.data_path):
            self.facts_df = pd.read_csv(self.data_path)
        else:
            self.facts_df = pd.DataFrame(columns=["id", "fact", "source"])
        return self.facts_df

    def save_facts(self):
        """Save facts to CSV file."""
        if self.facts_df is not None:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.facts_df.to_csv(self.data_path, index=False)

    def add_fact(self, fact: str, source: str):
        """Add a new fact to the database."""
        if self.facts_df is None:
            self.load_facts()

        # Get next ID
        next_id = self.facts_df["id"].max() + 1 if len(self.facts_df) > 0 else 1

        # Add new row
        new_row = pd.DataFrame([{
            "id": next_id,
            "fact": fact,
            "source": source
        }])

        self.facts_df = pd.concat([self.facts_df, new_row], ignore_index=True)

    def search_facts(self, query: str) -> pd.DataFrame:
        """Simple text search in facts."""
        if self.facts_df is None:
            self.load_facts()

        query_lower = query.lower()
        mask = self.facts_df["fact"].str.lower().str.contains(query_lower, na=False)
        return self.facts_df[mask]

    def get_all_facts(self) -> List[str]:
        """Get all facts as a list of strings."""
        if self.facts_df is None:
            self.load_facts()
        return self.facts_df["fact"].tolist()

    def get_facts_with_metadata(self) -> List[Dict]:
        """Get all facts with their metadata."""
        if self.facts_df is None:
            self.load_facts()
        return self.facts_df.to_dict("records")


# Example usage
if __name__ == "__main__":
    manager = FactBaseManager()
    facts_df = manager.load_facts()

    print(f"Loaded {len(facts_df)} facts from database")
    print("\nSample facts:")
    for _, row in facts_df.head(5).iterrows():
        print(f"- {row['fact'][:80]}...")
