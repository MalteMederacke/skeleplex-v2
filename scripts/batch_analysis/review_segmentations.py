"""Napari browser for segmentation QC.

Loads one zarr per sample — the best channel when a sample was segmented on
several (CHANNEL_PRIORITY), and nothing that has been curated out (see
``exclusions.py``). Each container was written by ``segment_batch.py`` and holds
the image, segmentation, and (when cropping was enabled) image_cropped and
segmentation_cropped. For speed the browser works on the CROPPED arrays when
present; saves are written back to segmentation_cropped AND scattered into the
full-size segmentation using the stored crop_bbox, so both stay consistent.

Widgets:
  Segmentation Browser  : Prev / Next through samples; re-run inference showing
                          ALL components; relabel without inference; save the
                          result; mark a sample bad quality
  Component Picker      : click a component in any labels layer -> Add/Remove
                          from a keep-list -> Apply to build a binary layer ->
                          Save that binary to the zarr

Re-run inference uses the checkpoint and roi_size belonging to the sample's own
channel, and the same normalisation ``segment_batch.py`` applies, so the re-run
reproduces the batch result rather than a slightly different one.

Usage:
    python review_segmentations.py
"""

import shutil
import sys
from pathlib import Path

import napari
import numpy as np
import zarr
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.morphology import label as sk_label

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    DEFAULT_VOXEL_SIZE_UM,
    checkpoint_for,
    inference_kwargs,
)
from _layout import find_segmented, image_key_in, rel  # noqa: E402
from exclusions import BAD_QUALITY_DIR  # noqa: E402
from normalization import normalize_image  # noqa: E402

# work on the cropped arrays when available (much faster to load / re-run)
PREFER_CROPPED = True

_RERUN_LAYER = "rerun_components"
_RERUN_BIN_LAYER = "rerun_binary"
_KEPT_LAYER = "kept_components"


def _chunks(shape, base=(64, 256, 256)):
    return tuple(min(b, s) for b, s in zip(base, shape))


def sample_keys(store, channel):
    """Return (image_key, seg_key, bbox) for this store.

    Prefers the cropped arrays; bbox (full-res crop bounds, or None) lets us
    keep cropped and full-size segmentations in sync on save.
    """
    bbox = None
    seg_attrs = store.attrs.get("segmentation", {})
    if isinstance(seg_attrs, dict):
        bbox = seg_attrs.get("crop_bbox")
    if PREFER_CROPPED and "image_cropped" in store:
        return "image_cropped", "segmentation_cropped", bbox
    return image_key_in(store, channel), "segmentation", bbox


def write_segmentation(store, binary, seg_key, bbox, full_shape):
    """Write ``binary`` to ``seg_key`` and keep its counterpart array in sync.

    ``binary`` is in the coordinate space of ``seg_key`` (cropped or full).
    """
    binary = binary.astype(np.uint8)
    if seg_key in store:
        del store[seg_key]
    store.create_array(seg_key, data=binary, chunks=_chunks(binary.shape))

    if bbox is None:
        return
    sl = tuple(slice(a, b) for a, b in bbox)
    if seg_key == "segmentation_cropped":
        full = np.zeros(full_shape, dtype=np.uint8)
        full[sl] = binary
        if "segmentation" in store:
            del store["segmentation"]
        store.create_array("segmentation", data=full, chunks=_chunks(full.shape))
    elif seg_key == "segmentation":
        crop = binary[sl]
        if "segmentation_cropped" in store:
            del store["segmentation_cropped"]
        store.create_array(
            "segmentation_cropped", data=crop, chunks=_chunks(crop.shape)
        )


class SegmentationBrowser(QWidget):
    def __init__(self, viewer: napari.Viewer, samples: list):
        super().__init__()
        self.viewer = viewer
        self.samples = samples
        self.index = 0
        self._rerun_result = None  # (labeled, binary) arrays from last re-run
        # per-sample context, set in load_current
        self.img_key = "image"
        self.seg_key = "segmentation"
        self.bbox = None
        self.full_shape = None

        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        btn_prev = QPushButton("← Prev")
        btn_prev.clicked.connect(self.prev)
        layout.addWidget(btn_prev)

        btn_next = QPushButton("Next →")
        btn_next.clicked.connect(self.next)
        layout.addWidget(btn_next)

        self.btn_rerun = QPushButton("Re-run segmentation (all components)")
        self.btn_rerun.clicked.connect(self.rerun_segmentation)
        layout.addWidget(self.btn_rerun)

        self.btn_relabel = QPushButton("Relabel components (no inference)")
        self.btn_relabel.clicked.connect(self.relabel_segmentation)
        layout.addWidget(self.btn_relabel)

        self.btn_save = QPushButton("Save re-run → zarr (overwrites segmentation)")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_rerun)
        layout.addWidget(self.btn_save)

        self.btn_bad = QPushButton(f"Mark bad quality (move to {BAD_QUALITY_DIR}/)")
        self.btn_bad.clicked.connect(self.mark_bad_quality)
        layout.addWidget(self.btn_bad)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()
        self.setLayout(layout)
        self.load_current()

    @property
    def sample(self):
        return self.samples[self.index]

    # ── rejection ────────────────────────────────────────────────────────────

    def mark_bad_quality(self):
        """Move the current sample's zarr AND its source image into bad_quality/.

        Both are moved: the zarr alone would be regenerated from the source on
        the next segment_batch.py run, silently undoing the rejection.
        BAD_QUALITY_DIR is one of the is_excluded() names, so the moved files
        are skipped by every stage of the pipeline from here on.
        """
        sample = self.sample
        moves = [
            (sample.zarr_path, sample.zarr_path.parent / BAD_QUALITY_DIR
             / sample.zarr_path.name)
        ]
        source = sample.source
        if source != sample.zarr_path and source.exists():
            moves.append((source, source.parent / BAD_QUALITY_DIR / source.name))

        detail = "\n".join(f"{src}\n  -> {dst}" for src, dst in moves)
        if QMessageBox.question(
            self, "Mark bad quality",
            f"Move {len(moves)} item(s) to {BAD_QUALITY_DIR}/?\n\n{detail}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        moved = []
        try:
            for src, dst in moves:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise FileExistsError(f"{dst} already exists")
                shutil.move(str(src), str(dst))
                moved.append((src, dst))
        except Exception as e:
            # put back whatever already moved, so we never leave a half-move
            for src, dst in reversed(moved):
                try:
                    shutil.move(str(dst), str(src))
                except Exception:
                    pass
            QMessageBox.critical(self, "Move failed", f"{e}\n\nNothing was moved.")
            return

        name = sample.zarr_path.name
        del self.samples[self.index]
        if not self.samples:
            for layer_name in list(self.viewer.layers):
                self.viewer.layers.remove(layer_name)
            self.label.setText("No samples left.")
            self.status.setText(f"Moved {len(moved)} item(s) to {BAD_QUALITY_DIR}/.")
            self.btn_bad.setEnabled(False)
            return
        self.index = min(self.index, len(self.samples) - 1)
        self.load_current()
        self.status.setText(
            f"Moved {len(moved)} item(s) to {BAD_QUALITY_DIR}/: {name}"
        )

    # ── navigation ───────────────────────────────────────────────────────────

    def _update_label(self):
        crop = " (cropped)" if self.img_key.endswith("_cropped") else ""
        self.label.setText(
            f"[{self.index + 1}/{len(self.samples)}]\n"
            f"{rel(self.sample.zarr_path)}{crop}\nchannel: {self.sample.channel}"
        )

    def load_current(self):
        self._rerun_result = None
        self.btn_save.setEnabled(False)
        self.status.setText("")

        # Clear only image/segmentation layers, keep any manually added ones
        for name in [
            self.img_key, self.seg_key, "image", "image_cropped",
            "segmentation", "segmentation_cropped",
            _RERUN_LAYER, _RERUN_BIN_LAYER,
        ]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        sample = self.sample
        store = zarr.open(str(sample.zarr_path))
        voxel_size = list(
            store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
        )
        scale = tuple(voxel_size)

        self.img_key, self.seg_key, self.bbox = sample_keys(store, sample.channel)
        self.full_shape = store[image_key_in(store, sample.channel)].shape

        img = np.asarray(store[self.img_key])
        if self.seg_key in store:
            seg = np.asarray(store[self.seg_key])
        else:
            seg = np.zeros(img.shape, dtype=np.uint8)

        self.viewer.add_image(
            img, name=self.img_key, scale=scale,
            colormap="gray", blending="additive",
        )
        self.viewer.add_labels(
            seg.astype(np.int32), name=self.seg_key, scale=scale
        )
        self.viewer.reset_view()
        self._update_label()

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def next(self):
        if self.index < len(self.samples) - 1:
            self.index += 1
            self.load_current()

    # ── inference / relabel / save ───────────────────────────────────────────

    def rerun_segmentation(self):
        from napari.qt.threading import thread_worker

        self.btn_rerun.setEnabled(False)
        self.status.setText("Running inference...")

        sample = self.sample
        store = zarr.open(str(sample.zarr_path))
        voxel_size = list(
            store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
        )
        scale = tuple(voxel_size)
        # Same normalisation the batch script applies, so the re-run reproduces
        # it; normalize_image is idempotent, so an already normalised array is
        # unchanged.
        img = normalize_image(np.asarray(store[self.img_key]))
        checkpoint = str(checkpoint_for(sample.channel))
        kwargs = inference_kwargs(sample.channel)

        @thread_worker
        def run():
            from inference import run_inference
            seg_raw = run_inference(img, checkpoint_path=checkpoint, **kwargs)
            binary = (seg_raw > 0.5).astype(np.uint8)
            labeled = sk_label(binary).astype(np.int32)
            return labeled, binary, scale

        def on_done(result):
            labeled, binary, scale = result
            self._rerun_result = (labeled, binary)
            for name in [_RERUN_LAYER, _RERUN_BIN_LAYER]:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self.viewer.add_labels(labeled, name=_RERUN_LAYER, scale=scale)
            n = labeled.max()
            self.status.setText(f"Re-run done: {n} component(s) found.")
            self.btn_rerun.setEnabled(True)
            self.btn_save.setEnabled(True)

        def on_error(exc):
            self.status.setText(f"ERROR: {exc}")
            self.btn_rerun.setEnabled(True)

        worker = run()
        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()

    def relabel_segmentation(self):
        """Connected-component label the current binary segmentation (no
        inference) so individual components can be picked in the Component
        Picker."""
        store = zarr.open(str(self.sample.zarr_path))
        voxel_size = list(
            store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
        )
        scale = tuple(voxel_size)

        # Prefer the (possibly edited) segmentation layer in the viewer,
        # falling back to what's stored on disk.
        if self.seg_key in self.viewer.layers:
            seg = np.asarray(self.viewer.layers[self.seg_key].data)
        else:
            seg = np.asarray(store[self.seg_key])

        binary = (seg > 0).astype(np.uint8)
        labeled = sk_label(binary).astype(np.int32)

        self._rerun_result = (labeled, binary)
        for name in [_RERUN_LAYER, _RERUN_BIN_LAYER]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.viewer.add_labels(labeled, name=_RERUN_LAYER, scale=scale)
        self.status.setText(
            f"Relabeled: {labeled.max()} component(s). Pick from '{_RERUN_LAYER}'."
        )
        self.btn_save.setEnabled(True)

    def save_rerun(self):
        if self._rerun_result is None:
            return
        _, binary = self._rerun_result
        store = zarr.open(str(self.sample.zarr_path), mode="r+")
        write_segmentation(store, binary, self.seg_key, self.bbox, self.full_shape)
        self.status.setText(f"Saved re-run binary → '{self.seg_key}' (+ counterpart).")
        self.btn_save.setEnabled(False)
        # Reload to show the saved version
        self.load_current()


class ComponentPicker(QWidget):
    """Pick components from any labels layer by clicking, build a binary mask."""

    def __init__(self, viewer: napari.Viewer, browser: SegmentationBrowser):
        super().__init__()
        self.viewer = viewer
        self.browser = browser
        self._keep: set[int] = set()
        self._watched_layer = None

        layout = QVBoxLayout()

        # Layer selector
        layout.addWidget(QLabel("Labels layer to pick from:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._watch_layer)
        layout.addWidget(self.layer_combo)

        btn_refresh = QPushButton("Refresh layer list")
        btn_refresh.clicked.connect(self._refresh_layers)
        layout.addWidget(btn_refresh)

        # Current selection indicator
        self.sel_label = QLabel("Selected component: —")
        layout.addWidget(self.sel_label)

        # Add / remove buttons
        row = QHBoxLayout()
        btn_add = QPushButton("Add ✓")
        btn_add.clicked.connect(self._add_selected)
        row.addWidget(btn_add)
        btn_remove = QPushButton("Remove ✗")
        btn_remove.clicked.connect(self._remove_selected)
        row.addWidget(btn_remove)
        layout.addLayout(row)

        # Keep list
        layout.addWidget(QLabel("Keep list:"))
        self.keep_list = QListWidget()
        self.keep_list.setMaximumHeight(160)
        layout.addWidget(self.keep_list)

        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear)
        layout.addWidget(btn_clear)

        # Apply / save
        btn_apply = QPushButton("Apply → new layer")
        btn_apply.clicked.connect(self._apply)
        layout.addWidget(btn_apply)

        self.btn_save_kept = QPushButton("Save kept binary → zarr")
        self.btn_save_kept.setEnabled(False)
        self.btn_save_kept.clicked.connect(self._save_kept)
        layout.addWidget(self.btn_save_kept)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()
        self.setLayout(layout)

        # Populate layer list once viewer is ready
        viewer.layers.events.inserted.connect(lambda _: self._refresh_layers())
        viewer.layers.events.removed.connect(lambda _: self._refresh_layers())
        self._refresh_layers()

    def _refresh_layers(self):
        current = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if hasattr(layer, "selected_label"):  # Labels layers only
                self.layer_combo.addItem(layer.name)
        # Restore previous selection if still present
        idx = self.layer_combo.findText(current)
        if idx >= 0:
            self.layer_combo.setCurrentIndex(idx)
        elif self.layer_combo.count() > 0:
            self.layer_combo.setCurrentIndex(self.layer_combo.count() - 1)
        self.layer_combo.blockSignals(False)
        self._watch_layer(self.layer_combo.currentText())

    def _watch_layer(self, name: str):
        # Disconnect old layer
        if self._watched_layer is not None:
            try:
                self._watched_layer.events.selected_label.disconnect(self._on_selected)
            except Exception:
                pass
            self._watched_layer = None

        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.events.selected_label.connect(self._on_selected)
            self._watched_layer = layer
            self._on_selected()

    def _on_selected(self, *_):
        if self._watched_layer is None:
            return
        lbl = self._watched_layer.selected_label
        self.sel_label.setText(f"Selected component: {lbl}")

    def _current_label(self):
        if self._watched_layer is None:
            return None
        return self._watched_layer.selected_label

    def _add_selected(self):
        lbl = self._current_label()
        if lbl is None or lbl == 0:
            return
        self._keep.add(lbl)
        self._refresh_list()

    def _remove_selected(self):
        lbl = self._current_label()
        if lbl is None:
            return
        self._keep.discard(lbl)
        self._refresh_list()

    def _clear(self):
        self._keep.clear()
        self._refresh_list()

    def _refresh_list(self):
        self.keep_list.clear()
        for k in sorted(self._keep):
            self.keep_list.addItem(str(k))

    def _apply(self):
        name = self.layer_combo.currentText()
        if not name or name not in self.viewer.layers:
            self.status.setText("No labels layer selected.")
            return
        if not self._keep:
            self.status.setText("Keep list is empty.")
            return

        layer = self.viewer.layers[name]
        data = np.asarray(layer.data)
        binary = np.zeros(data.shape, dtype=np.int32)
        for lbl in self._keep:
            binary[data == lbl] = lbl

        if _KEPT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_KEPT_LAYER)
        self.viewer.add_labels(binary, name=_KEPT_LAYER, scale=layer.scale)
        self.status.setText(f"Applied {len(self._keep)} component(s) → '{_KEPT_LAYER}'.")
        self.btn_save_kept.setEnabled(True)

    def _save_kept(self):
        if _KEPT_LAYER not in self.viewer.layers:
            self.status.setText("Apply selection first.")
            return
        kept_data = np.asarray(self.viewer.layers[_KEPT_LAYER].data)
        binary = (kept_data > 0).astype(np.uint8)

        browser = self.browser
        store = zarr.open(str(browser.sample.zarr_path), mode="r+")
        write_segmentation(
            store, binary, browser.seg_key, browser.bbox, browser.full_shape
        )
        self.status.setText(f"Saved kept binary → '{browser.seg_key}'.")
        self.btn_save_kept.setEnabled(False)
        browser.load_current()


def main():
    samples = find_segmented()
    if not samples:
        print("No segmentation zarrs found. Run segment_batch.py first.")
        sys.exit(1)
    print(f"Found {len(samples)} samples.")

    viewer = napari.Viewer(title="Segmentation QC")
    browser = SegmentationBrowser(viewer, samples)
    picker = ComponentPicker(viewer, browser)
    viewer.window.add_dock_widget(browser, name="Segmentation Browser", area="right")
    viewer.window.add_dock_widget(picker, name="Component Picker", area="right")
    napari.run()


if __name__ == "__main__":
    main()
