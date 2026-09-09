import runpy
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")
pytest.importorskip("rasterio")

from rangeshift.app_cli import main as app_launcher
from rangeshift.streamlit_app import main as packaged_app


def test_packaged_app_and_launcher_import() -> None:
    assert callable(packaged_app)
    assert callable(app_launcher)


def test_repository_streamlit_entrypoint_uses_packaged_app() -> None:
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    namespace = runpy.run_path(str(app_path), run_name="rangeshift_streamlit_test")
    assert namespace["main"] is packaged_app
