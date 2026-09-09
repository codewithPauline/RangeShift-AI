from pathlib import Path
import runpy

import pytest


pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")
pytest.importorskip("rasterio")


def test_streamlit_app_imports_without_starting_server() -> None:
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    namespace = runpy.run_path(str(app_path), run_name="rangeshift_streamlit_test")
    assert "main" in namespace
    assert callable(namespace["main"])
