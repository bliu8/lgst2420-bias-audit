import json
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def canned_results():
    with open(FIXTURES / "canned_responses.json") as f:
        return json.load(f)


@pytest.fixture
def sample_results_path():
    return FIXTURES / "sample_results.json"


@pytest.fixture
def sample_probes_csv():
    return FIXTURES / "sample_probes.csv"
