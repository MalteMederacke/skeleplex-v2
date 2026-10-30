"""QSS styles for the application.

Colors come from the palette of the QApplication via palette(...), so that the
styles follow the theme. See make_palette.
"""

from qtpy.QtGui import QColor, QPalette

from skeleplex.app._constants import WindowColors


def make_palette(colors: WindowColors) -> QPalette:
    """Make the application palette for the colors of a theme.

    Parameters
    ----------
    colors : WindowColors
        The colors of the window.

    Returns
    -------
    QPalette
        The palette to set on the QApplication.
    """
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, colors.window),
        (QPalette.ColorRole.WindowText, colors.text),
        (QPalette.ColorRole.Base, colors.base),
        (QPalette.ColorRole.AlternateBase, colors.group),
        (QPalette.ColorRole.Text, colors.text),
        (QPalette.ColorRole.ToolTipBase, colors.group),
        (QPalette.ColorRole.ToolTipText, colors.text),
        (QPalette.ColorRole.PlaceholderText, colors.disabled_text),
        (QPalette.ColorRole.Button, colors.button),
        (QPalette.ColorRole.ButtonText, colors.text),
        (QPalette.ColorRole.Mid, colors.strong_button),
        (QPalette.ColorRole.Dark, colors.border),
        (QPalette.ColorRole.Highlight, colors.highlight),
        (QPalette.ColorRole.HighlightedText, "#FFFFFF"),
        (QPalette.ColorRole.Link, colors.highlight),
    ):
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(
            QPalette.ColorGroup.Disabled, role, QColor(colors.disabled_text)
        )
    return palette


def shade_color(color: str, factor: float) -> str:
    """Scale the brightness of a hex color.

    Parameters
    ----------
    color : str
        The hex color, e.g. "#b7e2d8".
    factor : float
        The factor to multiply each channel with.

    Returns
    -------
    str
        The scaled hex color.
    """
    red, green, blue, _ = QColor(color).getRgb()
    return QColor(
        round(red * factor), round(green * factor), round(blue * factor)
    ).name()


# style for the flat group box title
# accent_color is the color of the title bar.
# background_color is the color of the group box.
FLAT_TITLE_STYLE = """
QLabel {{
    qproperty-alignment: AlignCenter;
	border: 1px solid {accent_color};
	background-color: {accent_color};
	padding: 3px 0px;
	color: palette(window-text);
	max-height: 20px;
        font-size: 12px;
}}
"""

FLAT_BUTTON_STYLE = """
QPushButton {{
    qproperty-alignment: AlignCenter;
	border: 1px solid {accent_color};
	background-color: {accent_color};
	padding: 5px 0px;
	color: black;
	max-height: 25px;
        font-size: 14px;
}}
"""

FLAT_FRAME_STYLE = """

QPushButton {
    border: None;
    background-color: palette(mid);
    color: palette(button-text);
    padding-left: 5px;
    padding-right: 5px;
    padding-top: 2px;
    padding-bottom: 2px;
}

"""

DOCK_WIDGET_STYLE = """
QDockWidget {
    border: None
}
QDockWidget::title {
    background: palette(button);
    padding: 2px;
    text-align: center;
}

QDockWidget::close-button, QDockWidget::float-button {
    border: 0px solid transparent;
    background: palette(button);
    padding: 0px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background: palette(mid);
}
"""
