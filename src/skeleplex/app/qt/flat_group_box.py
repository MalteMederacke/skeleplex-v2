"""A flat-styled group box widget."""

from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from skeleplex.app.qt.styles import (
    FLAT_FRAME_STYLE,
    FLAT_TITLE_STYLE,
    shade_color,
)


class FlatVFrame(QFrame):
    """A flat-styled frame widget with a vertical layout.

    Parameters
    ----------
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        # set the styling
        self.setStyleSheet(FLAT_FRAME_STYLE)

        self.setLayout(QVBoxLayout())

    def add_widget(self, widget: QWidget):
        """Add a widget to the frame.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.layout().addWidget(widget)


class FlatHFrame(QFrame):
    """A flat-styled frame widget with a horizontal layout.

    Parameters
    ----------
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        # set the styling
        self.setStyleSheet(FLAT_FRAME_STYLE)

        self.setLayout(QHBoxLayout())

    def add_widget(self, widget: QWidget):
        """Add a widget to the frame.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.layout().addWidget(widget)


class FlatVGroupBox(QWidget):
    """A flat-styled group box widget with a vertical layout.

    Parameters
    ----------
    title : str
        The title of the group box.
        The default is "".
    accent_color : str
        The accent color for the group box. This is
        used for the title bar and other accents.
        Color should be a hex string.
        The default is "#b7e2d8".
    background_color : str
        The background color for the group box.
        Color should be a hex string.
        The default is the group color of the theme.
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(
        self,
        title: str = "",
        accent_color: str = "#b7e2d8",
        background_color: str = "palette(alternate-base)",
        collapsible: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)

        # store the collapsible property
        self.collapsible = collapsible

        # set the background color
        self.setStyleSheet(f"background-color: {background_color};")

        self.title_widget = QLabel(title, parent=self)
        self._accent_color = accent_color
        self.set_title_shade(1.0)

        # add an callback to toggle the collapsed state when
        # the title is clicked.
        self.title_widget.mousePressEvent = self._toggle_collapsed

        self.frame = FlatVFrame(parent=self)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title_widget)
        layout.addWidget(self.frame)
        layout.addStretch()
        self.setLayout(layout)

    def set_title_shade(self, shade: float) -> None:
        """Scale the brightness of the accent color of the title.

        Parameters
        ----------
        shade : float
            The factor applied to the accent color.
        """
        self.title_widget.setStyleSheet(
            FLAT_TITLE_STYLE.format(accent_color=shade_color(self._accent_color, shade))
        )

    def add_widget(self, widget: QWidget):
        """Add a widget to the group box.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.frame.add_widget(widget)

    def _toggle_collapsed(self, event=None):
        """Toggle the collapsed state of the group box.

        This is used as a callback for when the title is clicked.
        """
        if not self.collapsible:
            # do nothing if not collapsible
            return
        if self.frame.isVisible():
            self.frame.setVisible(False)
            self.frame.setMaximumHeight(0)
            title_height = self.title_widget.sizeHint().height() + 10
            self.setMaximumHeight(title_height)
        else:
            self.frame.setVisible(True)
            title_height = self.title_widget.sizeHint().height() + 10
            frame_height = self.frame.sizeHint().height() + 10
            self.frame.setMaximumHeight(frame_height)
            self.setMaximumHeight(title_height + frame_height)


class FlatHGroupBox(QWidget):
    """A flat-styled group box widget with a horizontal layout.

    Parameters
    ----------
    title : str
        The title of the group box.
        The default is "".
    accent_color : str
        The accent color for the group box. This is
        used for the title bar and other accents.
        Color should be a hex string.
        The default is "#b7e2d8".
    background_color : str
        The background color for the group box.
        Color should be a hex string.
        The default is the group color of the theme.
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(
        self,
        title: str = "",
        accent_color: str = "#b7e2d8",
        background_color: str = "palette(alternate-base)",
        collapsible: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)

        # store the collapsible property
        self.collapsible = collapsible

        # set the background color
        self.setStyleSheet(f"background-color: {background_color};")

        self.title_widget = QLabel(title, parent=self)
        self._accent_color = accent_color
        self.set_title_shade(1.0)

        # add an callback to toggle the collapsed state when
        # the title is clicked.
        self.title_widget.mousePressEvent = self._toggle_collapsed

        self.frame = FlatHFrame(parent=self)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title_widget)
        layout.addWidget(self.frame)
        layout.addStretch()
        self.setLayout(layout)

    def set_title_shade(self, shade: float) -> None:
        """Scale the brightness of the accent color of the title.

        Parameters
        ----------
        shade : float
            The factor applied to the accent color.
        """
        self.title_widget.setStyleSheet(
            FLAT_TITLE_STYLE.format(accent_color=shade_color(self._accent_color, shade))
        )

    def add_widget(self, widget: QWidget):
        """Add a widget to the group box.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.frame.add_widget(widget)

    def _toggle_collapsed(self, event=None):
        """Toggle the collapsed state of the group box.

        This is used as a callback for when the title is clicked.
        """
        if not self.collapsible:
            # do nothing if not collapsible
            return
        if self.frame.isVisible():
            self.frame.setVisible(False)
            self.frame.setMaximumHeight(0)
            title_height = self.title_widget.sizeHint().height() + 10
            self.setMaximumHeight(title_height)
        else:
            self.frame.setVisible(True)
            title_height = self.title_widget.sizeHint().height() + 10
            frame_height = self.frame.sizeHint().height() + 10
            self.frame.setMaximumHeight(frame_height)
            self.setMaximumHeight(title_height + frame_height)
