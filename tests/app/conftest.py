"""Configuration for testing the viewer application."""

import importlib.util

# cellier is only installed for python >= 3.12 (see pyproject.toml)
collect_ignore_glob = [] if importlib.util.find_spec("cellier") else ["test_*.py"]
