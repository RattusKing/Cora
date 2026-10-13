import re
from pathlib import Path

import cora


def test_versions_agree():
    """One version everywhere: pyproject, the package, and the API. (No tomllib: it is 3.11+.)"""
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    assert cora.__version__ == match.group(1)
    from cora.api import create_app

    assert create_app(db_path=":memory:").version == cora.__version__
