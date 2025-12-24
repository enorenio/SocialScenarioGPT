"""
RocStories scenario loader for SIA-LLM experiments.
Provides utilities to load, sample, and iterate over scenarios.
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any

import pandas as pd


@dataclass
class Scenario:
    """A single scenario from RocStories."""
    story_id: str
    title: str
    description: str  # Combined sentences
    sentences: List[str]  # Individual sentences
    source: str  # 'spring2016' or 'winter2017'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RocStoriesLoader:
    """
    Loader for RocStories dataset.

    Usage:
        loader = RocStoriesLoader()

        # Get specific scenarios
        scenarios = loader.sample(n=20, seed=42)

        # Iterate over all
        for scenario in loader:
            print(scenario.title)
    """

    def __init__(
        self,
        dataset_dir: str = "Dataset",
        curated_file: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.curated_file = curated_file
        self._scenarios: Optional[List[Scenario]] = None
        self._df: Optional[pd.DataFrame] = None

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw RocStories CSVs."""
        if self._df is not None:
            return self._df

        dfs = []

        spring_file = self.dataset_dir / "ROCStories__spring2016.csv"
        if spring_file.exists():
            df = pd.read_csv(spring_file)
            df['source'] = 'spring2016'
            dfs.append(df)

        winter_file = self.dataset_dir / "ROCStories_winter2017.csv"
        if winter_file.exists():
            df = pd.read_csv(winter_file)
            df['source'] = 'winter2017'
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No RocStories files found in {self.dataset_dir}. "
                "Expected ROCStories__spring2016.csv and/or ROCStories_winter2017.csv"
            )

        self._df = pd.concat(dfs, ignore_index=True)
        return self._df

    def _row_to_scenario(self, row: pd.Series) -> Scenario:
        """Convert a DataFrame row to a Scenario object."""
        sentences = [
            row['sentence1'],
            row['sentence2'],
            row['sentence3'],
            row['sentence4'],
            row['sentence5'],
        ]
        description = " ".join(sentences)

        return Scenario(
            story_id=str(row['storyid']),
            title=row['storytitle'],
            description=description,
            sentences=sentences,
            source=row.get('source', 'unknown'),
        )

    def load_all(self) -> List[Scenario]:
        """Load all scenarios from the dataset."""
        if self._scenarios is not None:
            return self._scenarios

        df = self._load_raw_data()
        self._scenarios = [self._row_to_scenario(row) for _, row in df.iterrows()]
        return self._scenarios

    def sample(
        self,
        n: int = 20,
        seed: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[Scenario]:
        """
        Sample n scenarios from the dataset.

        Args:
            n: Number of scenarios to sample
            seed: Random seed for reproducibility
            source: Filter to specific source ('spring2016' or 'winter2017')

        Returns:
            List of Scenario objects
        """
        df = self._load_raw_data()

        if source:
            df = df[df['source'] == source]

        if n > len(df):
            n = len(df)

        sampled = df.sample(n=n, random_state=seed)
        return [self._row_to_scenario(row) for _, row in sampled.iterrows()]

    def get_by_title(self, title: str) -> Optional[Scenario]:
        """Get a scenario by its title."""
        df = self._load_raw_data()
        matches = df[df['storytitle'] == title]

        if matches.empty:
            return None

        return self._row_to_scenario(matches.iloc[0])

    def get_by_id(self, story_id: str) -> Optional[Scenario]:
        """Get a scenario by its story ID."""
        df = self._load_raw_data()
        matches = df[df['storyid'] == story_id]

        if matches.empty:
            return None

        return self._row_to_scenario(matches.iloc[0])

    def __iter__(self) -> Iterator[Scenario]:
        """Iterate over all scenarios."""
        return iter(self.load_all())

    def __len__(self) -> int:
        """Get total number of scenarios."""
        df = self._load_raw_data()
        return len(df)

    @property
    def total_count(self) -> int:
        """Get total number of stories available."""
        return len(self)

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        df = self._load_raw_data()

        return {
            "total_stories": len(df),
            "sources": df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            "avg_title_length": df['storytitle'].str.len().mean(),
            "avg_description_length": df.apply(
                lambda r: len(" ".join([
                    str(r['sentence1']), str(r['sentence2']),
                    str(r['sentence3']), str(r['sentence4']), str(r['sentence5'])
                ])), axis=1
            ).mean(),
        }


def save_curated_scenarios(
    scenarios: List[Scenario],
    output_file: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a curated list of scenarios to JSON.

    Args:
        scenarios: List of Scenario objects
        output_file: Path to output JSON file
        metadata: Optional metadata about the curation
    """
    output = {
        "metadata": metadata or {},
        "scenarios": [s.to_dict() for s in scenarios],
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


def load_curated_scenarios(input_file: str) -> List[Scenario]:
    """
    Load curated scenarios from JSON.

    Args:
        input_file: Path to curated JSON file

    Returns:
        List of Scenario objects
    """
    with open(input_file) as f:
        data = json.load(f)

    return [
        Scenario(**s) for s in data.get("scenarios", [])
    ]


if __name__ == "__main__":
    # Demo usage
    loader = RocStoriesLoader()

    print(f"Total stories: {loader.total_count}")
    print(f"Stats: {loader.stats()}")

    # Sample 5 stories
    print("\nSample scenarios:")
    for scenario in loader.sample(n=5, seed=42):
        print(f"  - {scenario.title}: {scenario.description[:80]}...")
