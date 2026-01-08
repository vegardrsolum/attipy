from pathlib import Path

import pytest
import attipy as ap

TEST_PATH = Path(__file__).parent


@pytest.fixture
def pva_data():
    fs = 10.24
    n = 1800.0 * fs  # 30 minutes
    return ap.pva_data(fs, n, type_="beating")
