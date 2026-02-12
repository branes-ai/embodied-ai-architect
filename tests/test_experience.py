"""Tests for the experience cache (SQLite-backed)."""

import pytest

from embodied_ai_architect.graphs.experience import DesignEpisode, ExperienceCache


@pytest.fixture()
def cache():
    """In-memory SQLite cache for testing."""
    c = ExperienceCache(db_path=":memory:")
    yield c
    c.close()


def _make_episode(**kwargs) -> DesignEpisode:
    defaults = dict(
        goal="Design drone SoC",
        use_case="delivery_drone",
        platform="drone",
        constraints={"max_power_watts": 5.0, "max_latency_ms": 33.3},
        architecture_chosen="SoC-KPU",
        hardware_selected="Stillwater KPU",
        ppa_achieved={"power_watts": 4.2, "latency_ms": 25.0},
        constraint_verdicts={"power": "PASS", "latency": "PASS"},
        outcome_score=1.0,
        iterations_used=2,
        key_decisions=["Selected KPU", "Applied INT8"],
    )
    defaults.update(kwargs)
    return DesignEpisode(**defaults)


class TestDesignEpisode:
    def test_defaults(self):
        ep = DesignEpisode()
        assert ep.episode_id  # auto-generated
        assert ep.timestamp
        assert ep.outcome_score == 0.0

    def test_compute_fingerprint(self):
        ep = _make_episode()
        fp = ep.compute_fingerprint()
        assert len(fp) == 16  # sha256 hex[:16]

    def test_fingerprint_deterministic(self):
        ep1 = _make_episode()
        ep2 = _make_episode()
        assert ep1.compute_fingerprint() == ep2.compute_fingerprint()

    def test_fingerprint_changes_with_constraints(self):
        ep1 = _make_episode(constraints={"max_power_watts": 5.0})
        ep2 = _make_episode(constraints={"max_power_watts": 10.0})
        assert ep1.compute_fingerprint() != ep2.compute_fingerprint()

    def test_serialization_round_trip(self):
        ep = _make_episode()
        data = ep.model_dump()
        restored = DesignEpisode(**data)
        assert restored.goal == ep.goal
        assert restored.outcome_score == ep.outcome_score


class TestExperienceCache:
    def test_save_and_load(self, cache):
        ep = _make_episode()
        eid = cache.save(ep)

        loaded = cache.load(eid)
        assert loaded is not None
        assert loaded.goal == "Design drone SoC"
        assert loaded.outcome_score == 1.0

    def test_load_nonexistent(self, cache):
        assert cache.load("nonexistent") is None

    def test_search_by_use_case(self, cache):
        cache.save(_make_episode(use_case="delivery_drone", outcome_score=1.0))
        cache.save(_make_episode(use_case="warehouse_amr", outcome_score=0.8))
        cache.save(_make_episode(use_case="delivery_drone", outcome_score=0.5))

        results = cache.search_similar(use_case="delivery_drone")
        assert len(results) == 2
        # Best score first
        assert results[0].outcome_score >= results[1].outcome_score

    def test_search_by_platform(self, cache):
        cache.save(_make_episode(platform="drone"))
        cache.save(_make_episode(platform="amr"))

        results = cache.search_similar(platform="drone")
        assert len(results) == 1
        assert results[0].platform == "drone"

    def test_search_by_fingerprint(self, cache):
        ep = _make_episode()
        ep.problem_fingerprint = ep.compute_fingerprint()
        cache.save(ep)

        results = cache.search_similar(fingerprint=ep.problem_fingerprint)
        assert len(results) == 1

    def test_search_empty(self, cache):
        results = cache.search_similar(use_case="nonexistent")
        assert results == []

    def test_search_limit(self, cache):
        for i in range(10):
            cache.save(_make_episode(use_case="test", outcome_score=i / 10))

        results = cache.search_similar(use_case="test", limit=3)
        assert len(results) == 3

    def test_list_episodes(self, cache):
        cache.save(_make_episode(use_case="drone"))
        cache.save(_make_episode(use_case="amr"))

        listing = cache.list_episodes()
        assert len(listing) == 2
        assert all("episode_id" in e for e in listing)
        assert all("use_case" in e for e in listing)

    def test_save_updates_existing(self, cache):
        ep = _make_episode(outcome_score=0.5)
        eid = cache.save(ep)

        # Update the score
        ep.outcome_score = 1.0
        cache.save(ep)

        loaded = cache.load(eid)
        assert loaded.outcome_score == 1.0

    def test_fingerprint_auto_computed(self, cache):
        ep = _make_episode()
        assert ep.problem_fingerprint == ""  # not yet computed

        cache.save(ep)

        loaded = cache.load(ep.episode_id)
        assert loaded.problem_fingerprint != ""
