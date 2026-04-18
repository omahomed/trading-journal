"""⌘K Command Palette — Streamlit custom component with bidirectional communication."""

import streamlit.components.v1 as components
import os

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_component_func = components.declare_component("cmdk", path=_COMPONENT_DIR)


def command_palette(pages, key=None):
    """Render the ⌘K command palette.

    Args:
        pages: list of dicts with 'label', 'group', 'color' keys
        key: optional Streamlit widget key

    Returns:
        Selected page label (str) or None
    """
    result = _component_func(pages=pages, key=key, default=None)
    return result
