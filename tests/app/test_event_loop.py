"""Tests that the qasync loop drives the asyncio tasks cellier schedules."""

import asyncio
import os

import pytest


@pytest.fixture
def qapp():
    """Provide a headless (offscreen) QApplication for loop tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_qasync_loop_drives_ensure_future(qapp):
    """A coroutine scheduled via ensure_future completes under a qasync loop.

    cellier's slicer submits work with ``asyncio.ensure_future`` on every
    reslice. Under ``%gui qt`` alone that task was scheduled on a parked loop
    and never ran (the Jupyter "buggy updates" bug). This asserts the qasync
    loop actually drives such tasks to completion.
    """
    import qasync

    event_loop = qasync.QEventLoop(qapp)
    asyncio.set_event_loop(event_loop)

    ran = {"value": False}

    async def _slice_like() -> None:
        await asyncio.sleep(0)
        ran["value"] = True

    async def _driver() -> None:
        await asyncio.ensure_future(_slice_like())

    with event_loop:
        event_loop.run_until_complete(_driver())

    assert ran["value"] is True
