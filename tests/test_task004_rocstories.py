"""
Tests for TASK-004: RocStories Dataset Preparation
Tests the scenario loader utility for loading and sampling RocStories.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils.scenario_loader import (
    RocStoriesLoader,
    Scenario,
    load_curated_scenarios,
    save_curated_scenarios,
)


def test_rocstories_loader_init():
    """Test RocStoriesLoader initialization."""
    loader = RocStoriesLoader(dataset_dir="Dataset")
    assert loader.dataset_dir == Path("Dataset")
    print("✓ RocStoriesLoader initializes correctly")


def test_rocstories_total_count():
    """Test that we can count total stories."""
    loader = RocStoriesLoader(dataset_dir="Dataset")
    count = loader.total_count
    assert count > 90000, f"Expected >90K stories, got {count}"
    print(f"✓ Total stories: {count:,}")


def test_rocstories_stats():
    """Test statistics calculation."""
    loader = RocStoriesLoader(dataset_dir="Dataset")
    stats = loader.stats()

    assert "total_stories" in stats
    assert "sources" in stats
    assert stats["total_stories"] > 90000
    assert "spring2016" in stats["sources"] or "winter2017" in stats["sources"]
    print(f"✓ Stats: {stats}")


def test_rocstories_sample():
    """Test sampling scenarios."""
    loader = RocStoriesLoader(dataset_dir="Dataset")

    # Sample with seed for reproducibility
    samples = loader.sample(n=5, seed=42)

    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, Scenario)
        assert s.story_id
        assert s.title
        assert s.description
        assert len(s.sentences) == 5
        assert s.source in ["spring2016", "winter2017"]

    # Verify reproducibility
    samples2 = loader.sample(n=5, seed=42)
    assert [s.story_id for s in samples] == [s.story_id for s in samples2]

    print(f"✓ Sampling works, first title: {samples[0].title}")


def test_rocstories_get_by_title():
    """Test getting scenario by title."""
    loader = RocStoriesLoader(dataset_dir="Dataset")

    # Known title from the dataset
    scenario = loader.get_by_title("Overweight Kid")

    assert scenario is not None
    assert scenario.title == "Overweight Kid"
    assert "Dan" in scenario.description

    # Non-existent title
    missing = loader.get_by_title("This Title Does Not Exist XYZ123")
    assert missing is None

    print(f"✓ get_by_title works: {scenario.title}")


def test_rocstories_get_by_id():
    """Test getting scenario by ID."""
    loader = RocStoriesLoader(dataset_dir="Dataset")

    # First get a scenario to find its ID
    samples = loader.sample(n=1, seed=123)
    story_id = samples[0].story_id

    # Now retrieve by ID
    scenario = loader.get_by_id(story_id)

    assert scenario is not None
    assert scenario.story_id == story_id

    print(f"✓ get_by_id works: {scenario.story_id}")


def test_load_curated_scenarios():
    """Test loading curated scenarios from JSON."""
    curated_path = "data_utils/rocstories_scenarios.json"

    if not Path(curated_path).exists():
        print("⚠ Skipping: curated scenarios file not found")
        return

    scenarios = load_curated_scenarios(curated_path)

    assert len(scenarios) > 0
    assert all(isinstance(s, Scenario) for s in scenarios)

    print(f"✓ Loaded {len(scenarios)} curated scenarios")


def test_scenario_to_dict():
    """Test Scenario serialization."""
    scenario = Scenario(
        story_id="test-123",
        title="Test Story",
        description="This is a test.",
        sentences=["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4.", "Sentence 5."],
        source="test",
    )

    d = scenario.to_dict()

    assert d["story_id"] == "test-123"
    assert d["title"] == "Test Story"
    assert d["description"] == "This is a test."
    assert len(d["sentences"]) == 5

    print("✓ Scenario.to_dict() works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-004: RocStories Dataset Preparation Tests")
    print("=" * 60)

    test_rocstories_loader_init()
    test_rocstories_total_count()
    test_rocstories_stats()
    test_rocstories_sample()
    test_rocstories_get_by_title()
    test_rocstories_get_by_id()
    test_load_curated_scenarios()
    test_scenario_to_dict()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
