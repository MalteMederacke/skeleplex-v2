"""Qt implementation of the main window for the application."""

from app_model import Application
from app_model.backends.qt import QModelMainWindow
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDockWidget, QTabBar, QWidget

from skeleplex.app._constants import ViewerTheme
from skeleplex.app.qt import AppControlsDock
from skeleplex.app.qt.app_controls import make_scroll_area
from skeleplex.app.qt.flat_group_box import FlatHGroupBox, FlatVGroupBox
from skeleplex.app.qt.main_viewer import MainViewerWidget
from skeleplex.app.qt.styles import DOCK_WIDGET_STYLE, make_palette

MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 600

#: The size the window opens with, if it fits on the screen.
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 1000

#: The width of the side panels the window opens with, if they can be that narrow.
DEFAULT_PANEL_WIDTH = 280

MAIN_WINDOW_STYLE = (
    """
* {
    font-size: 11px;
}

QToolBar {
    background: palette(window);
    border: none;
}

QTabBar::tab {
    padding: 3px 6px;
}

/* The value box of a slider hides its buttons, but they still take up space. */
QDoubleSpinBox[buttonSymbols="2"]::up-button,
QDoubleSpinBox[buttonSymbols="2"]::down-button {
    subcontrol-origin: border;
    width: 0px;
    height: 0px;
    border: 0px;
    image: none;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 11px;
    height: 11px;
    border: 1px solid palette(dark);
    background: palette(base);
}

QCheckBox::indicator {
    border-radius: 2px;
}

QRadioButton::indicator {
    border-radius: 6px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: palette(highlight);
}
"""
    + DOCK_WIDGET_STYLE
)


class MainWindow(QModelMainWindow):
    """Qt + app-model implementation of the main window for the application."""

    def __init__(self, app: Application):
        super().__init__(app)

        # Fusion follows the palette for all controls and is more compact than
        # the native styles.
        QApplication.setStyle("Fusion")
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        # self.tool_bar = self.addModelToolBar(MenuId.FILE, exclude={CommandId.OPEN})
        # self.tool_bar.setStyleSheet("background: white;")

        self._main_viewer_widget = MainViewerWidget(parent=self)
        self.setCentralWidget(self._main_viewer_widget)

        # set the minimum window size - app will launch with this size.
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # Create the app controls as a dock widget (left)
        self._create_app_controls()

        self._set_default_size()

    def _set_default_size(self) -> None:
        """Set the size of the window on opening, so that it fits on the screen."""
        available = self.screen().availableGeometry()
        self.resize(
            min(DEFAULT_WINDOW_WIDTH, available.width()),
            min(DEFAULT_WINDOW_HEIGHT, available.height()),
        )

    def fit_panels(self) -> None:
        """Make the side panels narrow, to leave the space to the canvas."""
        docks = self.findChildren(QDockWidget)
        self.resizeDocks(
            docks, [DEFAULT_PANEL_WIDTH] * len(docks), Qt.Orientation.Horizontal
        )
        for tab_bar in self.findChildren(QTabBar):
            tab_bar.setElideMode(Qt.TextElideMode.ElideNone)

        # the sizes of a window that is not laid out yet would be overridden
        QTimer.singleShot(0, self._fit_panel_rows)

    def _fit_panel_rows(self) -> None:
        """Give each row of tabs on the right the height of its tallest widget."""
        rows: list[QDockWidget] = []
        heights: list[int] = []
        visited: set[QDockWidget] = set()
        for dock in self._auxiliary_docks.values():
            if (
                dock in visited
                or self.dockWidgetArea(dock) != Qt.DockWidgetArea.RightDockWidgetArea
            ):
                continue
            row = [dock, *self.tabifiedDockWidgets(dock)]
            visited.update(row)
            rows.append(dock)
            heights.append(max(d.widget().widget().sizeHint().height() for d in row))
        self.resizeDocks(rows, heights, Qt.Orientation.Vertical)

    def apply_theme(self, theme: ViewerTheme) -> None:
        """Set the colors of the window.

        Parameters
        ----------
        theme : ViewerTheme
            The theme to apply.
        """
        QApplication.instance().setPalette(make_palette(theme.window))
        for group_box_type in (FlatVGroupBox, FlatHGroupBox):
            for group_box in self.findChildren(group_box_type):
                group_box.set_title_shade(theme.window.title_shade)

        # the palette(...) colors of a style sheet are resolved when it is set
        for widget in (self, *self.findChildren(QWidget)):
            style_sheet = widget.styleSheet()
            if style_sheet:
                widget.setStyleSheet(style_sheet)

    def add_auxiliary_widget(
        self,
        widget: QWidget,
        name: str = "",
        area: str = "right",
        tabify_with: str | None = None,
    ):
        """Add a widget to the auxiliary views dock.

        If tabify_with names a widget that was added before, the new widget
        is placed in a tab behind it instead of taking its own space.
        """
        dock_widget = QDockWidget(name)
        dock_widget.setWidget(make_scroll_area(widget))
        if not hasattr(self, "_auxiliary_docks"):
            self._auxiliary_docks = {}
        self._auxiliary_docks[name] = dock_widget
        if tabify_with in self._auxiliary_docks:
            self.tabifyDockWidget(self._auxiliary_docks[tabify_with], dock_widget)
            # keep the first widget of the tabs on top
            self._auxiliary_docks[tabify_with].raise_()
            return
        dock_area = (
            Qt.DockWidgetArea.LeftDockWidgetArea
            if area == "left"
            else Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(dock_area, dock_widget)

    def _create_app_controls(self):
        self.app_controls = AppControlsDock(parent=self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.app_controls)

    def _set_main_viewer_widget(self, canvas_widget: QWidget):
        """Set the main viewer widget."""
        self._main_viewer_widget.addCanvasWidget(canvas_widget)
