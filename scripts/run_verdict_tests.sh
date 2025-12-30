#!/bin/bash
# Run verdict-first tools test suite
#
# Usage:
#   ./scripts/run_verdict_tests.sh              # Unit tests only (no API)
#   ./scripts/run_verdict_tests.sh --live       # Live API tests (smoke)
#   ./scripts/run_verdict_tests.sh --live-all   # Live API tests (all)
#   ./scripts/run_verdict_tests.sh --interactive # Interactive mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "======================================"
echo "Verdict-First Tools Test Suite"
echo "======================================"

# Check for API key if running live tests
check_api_key() {
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: ANTHROPIC_API_KEY not set"
        echo "Set your API key: export ANTHROPIC_API_KEY=your-key"
        exit 1
    fi
}

case "${1:-unit}" in
    --live|live)
        echo "Running smoke tests with live API..."
        check_api_key
        pytest tests/test_verdict_tools_cli.py::TestVerdictToolsLive::test_smoke -v -s
        ;;
    --live-all|live-all)
        echo "Running ALL tests with live API..."
        check_api_key
        pytest tests/test_verdict_tools_cli.py::TestVerdictToolsLive -v -s
        ;;
    --interactive|interactive)
        echo "Running interactive test mode..."
        check_api_key
        python tests/test_verdict_tools_cli.py
        ;;
    --unit|unit|"")
        echo "Running unit tests (no API required)..."
        pytest tests/test_verdict_tools_cli.py::TestVerdictToolsUnit -v
        ;;
    --help|help)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  --unit        Run unit tests only (default, no API required)"
        echo "  --live        Run smoke tests with live API"
        echo "  --live-all    Run all tests with live API"
        echo "  --interactive Run interactive test mode"
        echo "  --help        Show this help"
        echo ""
        echo "Environment:"
        echo "  ANTHROPIC_API_KEY  Required for live tests"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run '$0 --help' for usage"
        exit 1
        ;;
esac

echo ""
echo "Done!"
