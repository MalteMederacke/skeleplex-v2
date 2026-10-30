"""Constants for the application."""

from dataclasses import dataclass
from enum import Enum


class CommandId(str, Enum):
    """Command identifiers for the application."""

    OPEN = "skeleplex.open"
    CLOSE = "skeleplex.close"
    SAVE = "skeleplex.save"
    QUIT = "skeleplex.quit"
    LOAD_DATA = "skeleplex.load_data"
    LOOK_AT_SKELETON = "skeleplex.look_at_skeleton"
    PASTE_EDGE_SELECTION = "skeleplex.paste_edge_selection"
    PASTE_NODE_SELECTION = "skeleplex.paste_node_selection"

    UNDO = "skeleplex.undo"
    REDO = "skeleplex.redo"

    TOGGLE_THEME = "skeleplex.toggle_theme"

    def __str__(self) -> str:
        """String representation."""
        return self.value


class MenuId(str, Enum):
    """Menu identifiers for the application."""

    FILE = "skeleplex/file"
    EDIT = "skeleplex/edit"
    DATA = "skeleplex/data"
    VIEW = "skeleplex/view"

    def __str__(self) -> str:
        """String representation."""
        return self.value


@dataclass(frozen=True)
class WindowColors:
    """The colors of the window around the canvas. All colors are hex strings.

    Parameters
    ----------
    window : str
        The background of the window and the panels.
    group : str
        The background of the groups of controls within a panel.
    base : str
        The background of text inputs and lists.
    text : str
        The color of all text.
    button : str
        The background of standard buttons and of panel titles.
    strong_button : str
        The background of the buttons in the group boxes.
    border : str
        The color of borders.
    highlight : str
        The background of highlighted items, like the selected text.
    disabled_text : str
        The color of the text of disabled controls.
    title_shade : float
        The factor applied to the accent color of the group box titles.
    """

    window: str
    group: str
    base: str
    text: str
    button: str
    strong_button: str
    border: str
    highlight: str
    disabled_text: str
    title_shade: float


@dataclass(frozen=True)
class ViewerTheme:
    """The colors of the application. Scene colors are RGBA in [0, 1].

    Parameters
    ----------
    background : tuple[float, float, float, float]
        The color behind the scene.
    node_color : tuple[float, float, float, float]
        The color of the skeleton nodes.
    edge_color : tuple[float, float, float, float]
        The color of edges when no colormap has been applied.
    missing_edge_color : tuple[float, float, float, float]
        The color of edges without a value when coloring by an edge attribute.
    window : WindowColors
        The colors of the window around the canvas.
    """

    background: tuple[float, float, float, float]
    node_color: tuple[float, float, float, float]
    edge_color: tuple[float, float, float, float]
    missing_edge_color: tuple[float, float, float, float]
    window: WindowColors


THEMES: dict[str, ViewerTheme] = {
    "light": ViewerTheme(
        background=(1.0, 1.0, 1.0, 1.0),
        node_color=(0.0, 0.0, 0.0, 0.8),
        edge_color=(0.0, 0.0, 1.0, 1.0),
        missing_edge_color=(0.0, 0.0, 0.0, 1.0),
        window=WindowColors(
            window="#F5F6F7",
            group="#D5D6D7",
            base="#FFFFFF",
            text="#000000",
            button="#E5E6E8",
            strong_button="#95A7AE",
            border="#6B6F73",
            highlight="#3D7BD9",
            disabled_text="#808285",
            title_shade=1.0,
        ),
    ),
    "dark": ViewerTheme(
        background=(0.07, 0.07, 0.09, 1.0),
        node_color=(1.0, 1.0, 1.0, 0.8),
        edge_color=(0.35, 0.65, 1.0, 1.0),
        missing_edge_color=(1.0, 1.0, 1.0, 1.0),
        window=WindowColors(
            window="#1B1C21",
            group="#26272D",
            base="#3A3C44",
            text="#E4E5E7",
            button="#33353C",
            strong_button="#46535A",
            border="#55585F",
            highlight="#3D7BD9",
            disabled_text="#7A7C82",
            title_shade=0.4,
        ),
    ),
}
