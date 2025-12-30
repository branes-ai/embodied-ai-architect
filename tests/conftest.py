"""Pytest configuration for embodied-ai-architect tests."""

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--live-api",
        action="store_true",
        default=False,
        help="Run tests against live Anthropic API (requires ANTHROPIC_API_KEY)",
    )


@pytest.fixture
def live_api(request):
    """Check if live API testing is enabled."""
    return request.config.getoption("--live-api")
