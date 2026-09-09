"""Launch the packaged RangeShift Streamlit explorer."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch ``rangeshift.streamlit_app`` with Streamlit's CLI."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise ImportError(
            "The interactive explorer requires Streamlit. "
            "Install RangeShift with the 'app' and 'geo' extras."
        ) from exc

    app_path = Path(__file__).with_name("streamlit_app.py")
    passthrough = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app_path), *passthrough]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
