"""Convert a Nikon ND2 file into one ``ArrayDataset`` per stage position.

The same contract as the LIF and OME converters, read natively: no Bio-Formats, no JVM
and no OME in between. ``nd2`` reads each frame out of the file on demand, so the pixels
are never decoded into RAM or copied to a scratch store before upload, and the metadata
comes from NIS-Elements' own record of the run — which keeps what an OME translation
drops: each channel's modality, its laser lines and detection band, a point scanner's
PMT voltages, the camera, its exposure and binning, where the stage put every point of a
multipoint run, and when every frame was taken. See `nd2_metadata`.

Everything the file says about an image is pinned to a *coordinate*: facts true of the
whole image (NIS's record, the stand, the objective, the camera, the stage, when it was
acquired) hang off the anchor that pins no axis; facts true of one channel (its
histogram, modality, filter, detector, light path) hang off that channel's anchor.
Physical scale is an edge, not a property. An image shares the file's micrometer world
only where the file says where it sits along EVERY spatial axis it has: x and y from the
stage, and z for a stack from the per-plane focus NIS recorded (`world_offsets`).
Otherwise it is registered, calibrated, into a micrometer space of its own, and staged
there alone.

A multipoint run is one ND2 with a ``P`` axis. Each point is uploaded as a dataset of its
own and registered at its own stage position. Points whose fields touch (a grid NIS did
not stitch) share one mosaic scene; points that do not (wells of a plate) each get a scene
of their own — see `share_a_scene`. An image NIS stitched itself is one field.

Nothing about a scene is bootstrapped: every lens and layer is authored explicitly,
**one layer per channel**. The only exception is a packed 8-bit RGB pixel from a colour
camera — see `packs_one_pixel`.

Absent facts stay absent. An image whose pixels NIS did not calibrate is **not**
registered into the world: ``scale={x: 1.0}`` is not a neutral default but the claim that
one pixel is one micrometer. Such a dataset is staged in its own pixel grid instead.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import struct
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from arkitekt import App, Task, run
from mikro import Mikro, build_pyramid, canonical, mikro_service, scales_from, space_3d
from mikro.api.schema import (
    PlacementValidity,
    AcquisitionWindowInput,
    ArrayDataset,
    AxisAnchorInput,
    AxisType,
    Blending,
    ColorMap,
    CoordinateAnchorInput,
    File,
    LabelInput,
    OmeMetadataInput,
    OptikitStateInput,
    PreferredView,
    ProjectionMode,
    Scene,
    SourceFileInput,
    StageStateInput,
    ValueHistogramInput,
    ValueRelation,
)
from mikro.inputs.render import (
    RGB_PRIMARIES,
    color_for_wavelength,
    composite_graph,
    fallback_color,
    rgba,
)
from mikro.pyramid import axes_for

from converter_metadata import device, devices, setting
from logos import declare_logos, logo_of
from nd2_instruments import NikonAcquisition, acquisition_of
from nd2_metadata import (
    Calibration,
    FrameTable,
    NikonChannel,
    NikonScope,
    Position,
    ZStack,
    image_record,
    picture_metadata,
    read_calibration,
    read_channels,
    read_frame_table,
    read_positions,
    read_scope,
    z_stack,
)

logger = logging.getLogger(__name__)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.0.0"

app = App(
    "nd2-converter",
    __version__,
    description=(
        "Converts Nikon ND2 files into mikro datasets natively, one per stage position, "
        "with NIS-Elements' own metadata: modalities, laser lines, detection bands, "
        "camera, detectors, pinhole, stage positions and frame times."
    ),
    author="jhnnsrs",
)
app.service(mikro_service)
declare_logos(app)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The axes the world below is built over. Every scale and offset is derived by
# intersecting an image's own dims with this set: `register` raises if a scale or an
# offset names an axis the two spaces do not share.
WORLD_AXES = frozenset({"z", "y", "x"})

# The axis type for each name this converter can emit. Stated rather than inferred:
# the library's fallback calls every unrecognised name SPACE, which is right for
# z/y/x and silently wrong for anything else — and it matters twice over, because
# the pyramid coarsens exactly the SPACE axes.
AXIS_TYPES: dict[str, AxisType] = {
    "t": AxisType.TIME,
    "c": AxisType.CHANNEL,
    "z": AxisType.SPACE,
    "y": AxisType.SPACE,
    "x": AxisType.SPACE,
}

AXIS_LONG_NAMES = {
    "t": "time",
    "c": "channel",
    "z": "depth",
    "y": "height",
    "x": "width",
}

# `ValueHistogramInput.from_array` calls `np.asarray`, so it is the one place a lazy
# image is forced into memory. Above this many bytes per channel slice the histogram
# is computed from the finest pyramid level that fits instead.
HISTOGRAM_BUDGET_BYTES = 256 * 1024 * 1024

# Past this many layers a scene stops being useful: a 96-well run of four channels is
# 384. The world still holds every registration and outlives the run, so a subset can be
# staged later.
MAX_SCENE_LAYERS = 48

# How far apart two fields may be and still count as touching, as a fraction of the
# field: a grid acquired with a little overlap or a little gap is one mosaic, the wells
# of a plate are not.
MOSAIC_GAP_FRACTION = 0.1

# An automatic pyramid halves until its coarsest level fits a tile of this many pixels
# on its longer side: enough to open a 12k stitched image in one fetch, and no more
# levels than a 512-pixel field needs.
PYRAMID_TOP_PIXELS = 512

# The longer side of a scene snapshot, in pixels. A mosaic's tiles share it: each is
# rendered at its share of the side, so 48 tiles cost what one field does.
SNAPSHOT_PIXELS = 1024

READABLE_SUFFIXES = (".nd2",)

# What a channel is drawn in when NIS did not record a colour: its emission, mapped to
# roughly the colour the eye would have seen down the eyepiece.
COLOR_BY_EMISSION = (
    (400, (150, 0, 255)),
    (450, (75, 0, 255)),
    (490, (0, 130, 255)),
    (520, (0, 255, 200)),
    (560, (0, 255, 0)),
    (590, (255, 255, 0)),
    (625, (255, 150, 0)),
    (700, (255, 0, 0)),
)

FAR_RED = (255, 70, 70)


# ---------------------------------------------------------------------------
# The array
# ---------------------------------------------------------------------------


def file_array(f: Any) -> xr.DataArray:
    """The whole of an ND2 file, lazily: one frame per chunk, read on demand.

    ``to_dask`` reads a frame where it lies when a chunk is computed, so nothing is
    decoded up front and nothing is copied to a scratch file. Its default wrapper reopens
    the file for a computation that outlives the handle, but the converter keeps the
    handle open for the whole run anyway.

    Dims are nd2's, in stored order (``T P Z C Y X``, ``Y X S`` …), with singleton
    loops already dropped.
    """
    return xr.DataArray(f.to_dask(), dims=[str(d) for d in f.sizes])


def field_array(
    whole: xr.DataArray,
    position: int | None,
    rgb_channel: int | None,
    descending_z: bool,
) -> tuple[xr.DataArray, dict[str, str], bool]:
    """One point (and, for a multi-channel colour camera, one channel) as a named
    array, its extra axis types, and whether ``c`` is a packed sample axis.

    The same reduction the LIF converter makes: dims lower-cased, ``c`` kept even at
    extent 1 (the per-channel anchors key off it), any other singleton dropped, and
    anything left that is not t/c/z/y/x declared INDEX so the pyramid does not halve it.

    A colour camera's pixel is three samples — nd2's ``S``. With several channels of them
    (``C`` × ``S``) each channel is a picture of its own, and is converted as one
    (`rgb_channel`). A stack NIS acquired top-down (`descending_z`) is turned over, so
    that z grows with its index, as the scale registering it claims.
    """
    array = whole.isel(P=position) if position is not None else whole

    packed_samples = False
    if "S" in array.dims:
        if "C" in array.dims:
            if rgb_channel is None:
                raise ValueError(
                    f"{array.sizes['C']} channels of {array.sizes['S']} samples each: "
                    "convert them one channel at a time."
                )
            array = array.isel(C=rgb_channel)
        packed_samples = array.sizes["S"] in (3, 4)
        array = array.rename({"S": "c"})

    array = array.rename({d: str(d).lower() for d in array.dims if str(d) in "TCZYX"})

    if descending_z and "z" in array.dims:
        array = array.isel(z=slice(None, None, -1))

    if "c" not in array.dims:
        array = array.expand_dims("c")

    dropped = [
        d for d in array.dims if str(d) not in ("c", "y", "x") and array.sizes[d] == 1
    ]
    if dropped:
        array = array.squeeze(dropped, drop=True)

    extra_types = {str(d): "INDEX" for d in array.dims if str(d) not in AXIS_TYPES}

    return (
        xr.DataArray(array.data, dims=[str(d) for d in array.dims]),
        extra_types,
        packed_samples,
    )


def auto_levels(sizes: Mapping[str, int]) -> int:
    """How many pyramid levels an image needs: halvings until it fits `PYRAMID_TOP_PIXELS`.

    Including the base, so a 1024-pixel field is two levels and an 11k stitched image is
    six. A fixed count is wrong in both directions — levels below one tile are wasted
    uploads, and a stitched image stopped at a fixed depth still opens at megapixels.

    Never fewer than two: a one-level pyramid uploads with an empty ``scales`` list, which
    no other converter here sends — one coarser level costs a quarter of a small field.
    """
    side = max(int(sizes.get("y", 1)), int(sizes.get("x", 1)))
    levels = 1
    while side > PYRAMID_TOP_PIXELS or levels < 2:
        side = -(-side // 2)
        levels += 1
    return levels


def image_pyramid(
    data: xr.DataArray,
    levels: int,
    method: str,
    types: Mapping[str, str],
    scratch: str,
) -> list[xr.DataArray]:
    """The pyramid of a memory-mapped base: level 0 left mapped, level 1 persisted.

    mikro uploads each level as a separate computation, and a lazy level is chained all
    the way back to level 0 — so a pyramid left wholly lazy reads the base once per
    level, six passes over a 40 GB stitched image. mikro's own remedy (``materialize``)
    copies the whole base to a scratch Zarr first, in ``TMPDIR`` whatever
    ``MIKRO_SCRATCH_DIR`` says, which for a base that is already a random-access store
    is a full copy for nothing.

    The middle path: level 1 — a quarter of the base, an eighth for a z-stack — is
    written once into `scratch` (a directory the action owns, beside the download, and
    removes once this dataset is uploaded), and every coarser level is built from it. Two passes over the base in
    all: one for level 1, one for the upload of level 0 itself.
    """
    import dask.array as da

    pyramid = build_pyramid(
        data, levels=levels, method=method, types=types, materialize=False
    )
    if len(pyramid) <= 2:
        return pyramid

    path = tempfile.mkdtemp(prefix="level1-", dir=scratch)
    pyramid[1].data.to_zarr(path, overwrite=True)
    level1 = xr.DataArray(da.from_zarr(path), dims=pyramid[1].dims)
    return [
        pyramid[0],
        *build_pyramid(
            level1, levels=len(pyramid) - 1, method=method, types=types, materialize=False
        ),
    ]

# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def voxel_scale(
    calibration: Calibration, dims: Sequence[str]
) -> tuple[dict[str, float] | None, str | None]:
    """The per-axis factor from this dataset's indices into micrometers.

    Returns ``(None, reason)`` when NIS did not calibrate the pixels. Registering anyway
    would positively claim one pixel is one micrometer, at the wrong size next to
    genuinely calibrated images.

    A missing *axial* spacing is different: omitting ``z`` and passing ``z=1.0`` are the
    same claim, so there is no way to abstain. The assumption is recorded instead.
    """
    shared = [d for d in dims if d in WORLD_AXES]
    if calibration.x_um is None or calibration.y_um is None:
        return (
            None,
            "the pixels are not calibrated — not registered into a physical space",
        )
    note = None
    if "z" in shared and calibration.z_um is None:
        note = "z spacing unknown, registered as 1 um per plane"
    sizes = {"x": calibration.x_um, "y": calibration.y_um, "z": calibration.z_um}
    return {axis: sizes[axis] or 1.0 for axis in shared}, note


@dataclass(frozen=True)
class Orientation:
    """How the image's axes lie along the stage's: which stage axis grows with the
    column index and which with the row, and in which direction.

    NIS records the camera against the stage as a near-rotation (``cameraTransformationMatrix``,
    image y pointing up). On most of its cameras that is 180°: a column further right is
    a stage x further left. Placing fields by their stage centres with that ignored would
    mirror any grid of them. The world here is laid out the way the pixels are — x along
    columns, y along rows — and each stage centre is carried into it by the nearest
    axis-aligned orientation. What is left over (a degree or two of rotation) cannot be
    expressed by a scale-and-offset placement, and is recorded on the image anchor.
    """

    #: Stage axis (0 = x, 1 = y) and sign that a column step follows.
    column: tuple[int, int]
    #: Stage axis and sign that a row step follows.
    row: tuple[int, int]
    source: str

    def world_xy(self, stage_x: float, stage_y: float) -> tuple[float, float]:
        stage = (stage_x, stage_y)
        return (
            self.column[1] * stage[self.column[0]],
            self.row[1] * stage[self.row[0]],
        )


IDENTITY_ORIENTATION = Orientation((0, 1), (1, 1), "none recorded")


def orientation_of(scope: NikonScope) -> Orientation:
    """The orientation the file records: the pixel → stage matrix where NIS calibrated
    one, else the camera matrix, else none (stage axes taken as the image's)."""
    if scope.pixel_to_stage and len(scope.pixel_to_stage) == 6:
        m00, m01, _, m10, m11, _ = scope.pixel_to_stage
        # Columns: d(stage)/d(x) = (m00, m10); rows: d(stage)/d(row) = (m01, m11).
        return Orientation(
            _dominant(m00, m10), _dominant(m01, m11), "pixelToStageTransformationMatrix"
        )
    if scope.camera_matrix and len(scope.camera_matrix) == 4:
        a, b, c, d = scope.camera_matrix
        # stage_x = a·x + c·y_up, stage_y = b·x + d·y_up, and a row step is y_up = -1.
        return Orientation(
            _dominant(a, b), _dominant(-c, -d), "cameraTransformationMatrix"
        )
    return IDENTITY_ORIENTATION


def _dominant(along_x: float, along_y: float) -> tuple[int, int]:
    """The stage axis a vector mostly follows, and the sign it follows it with."""
    if abs(along_x) >= abs(along_y):
        return (0, 1 if along_x >= 0 else -1)
    return (1, 1 if along_y >= 0 else -1)


@dataclass(frozen=True)
class StageCentre:
    """Where the stage put the centre of one field, in world micrometers."""

    x: float
    y: float
    stage_x: float
    stage_y: float
    orientation: Orientation


def stage_centre(position: Position, orientation: Orientation) -> StageCentre | None:
    """The centre of this point's field in the world, or None when the stage was not
    read out (`nd2_metadata.stage_xy`). NIS records the centre of the field."""
    if position.x_um is None or position.y_um is None:
        return None
    x, y = orientation.world_xy(position.x_um, position.y_um)
    return StageCentre(x, y, position.x_um, position.y_um, orientation)


def stack_bottom(
    frames: FrameTable, position: int | None, stack: ZStack | None
) -> float | None:
    """The z of the lowest plane of this point's stack, in micrometers, or None.

    From the per-plane ``Z Coord`` NIS recorded, at the first timepoint: the only
    statement of where each plane was. The frame metadata's stage z is constant over a
    stack (the bottom plane in one file, the home plane in another), so it is not used.
    A run that drove its planes on a piezo it did not read out records no ``Z Coord``,
    and then the stack's place along z is not known.
    """
    if stack is None:
        return None
    coords: dict[str, int] = {"T": 0}
    if position is not None:
        coords["P"] = position
    seqs = frames.select(**coords)
    if not len(seqs):
        return None
    z = frames.z_um[seqs]
    if np.isnan(z).any() or not np.any(z):
        return None
    return float(np.min(z))


def world_offsets(
    centre: StageCentre | None,
    scale: Mapping[str, float],
    data: xr.DataArray,
    bottom_um: float | None,
) -> dict[str, float] | None:
    """Where pixel (0, 0, 0) sits in the shared world, or None when the file does not say.

    All or nothing: every spatial axis the image has must be placed — x and y from the
    stage, z from the stack's lowest plane. A space shared with other images is the claim
    that positions in it compare; an axis left at 0 would put the image where the
    instrument never said it was. Pixel centres, so a field of N pixels spans N - 1
    spacings between its first and last.
    """
    offsets: dict[str, float] = {}
    if centre is not None:
        for axis, value in (("x", centre.x), ("y", centre.y)):
            if axis in data.dims and axis in scale:
                offsets[axis] = value - (data.sizes[axis] - 1) * scale[axis] / 2
    if "z" in data.dims and bottom_um is not None:
        offsets["z"] = bottom_um
    spatial = {axis for axis in ("z", "y", "x") if axis in data.dims}
    return offsets if spatial <= offsets.keys() else None


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def frame_windows(
    frames: FrameTable, position: int | None, timepoints: int
) -> tuple[AcquisitionWindowInput | None, list[tuple[int, AcquisitionWindowInput]]]:
    """When this point was acquired: as a whole, and per timepoint.

    NIS stamps every frame. The whole runs from this point's first frame to its last; a
    timepoint spans its own frames — every plane and channel taken at it. Timepoints are
    only stated for a T longer than one, keyed by their index along it.
    """
    seqs = frames.select(P=position) if position is not None else np.arange(len(frames))
    times = frames.times[seqs]
    whole = span(times)
    per_t: list[tuple[int, AcquisitionWindowInput]] = []
    if timepoints > 1:
        # Within this point's frames only: a run's worth of timepoints, each a mask over
        # one point's frames rather than over the whole file.
        along_t = frames.axis("T")[seqs]
        for index in range(timepoints):
            window = span(times[along_t == index])
            if window is not None:
                per_t.append((index, window))
    return whole, per_t


def span(stamps: np.ndarray) -> AcquisitionWindowInput | None:
    """The first and last of these stamps, ignoring the missing ones; None when none is left."""
    stamps = np.asarray(stamps).ravel()
    stamps = stamps[~np.isnat(stamps)]
    if not stamps.size:
        return None
    return AcquisitionWindowInput(start=utc(stamps.min()), end=utc(stamps.max()))


def utc(stamp: np.datetime64) -> datetime.datetime:
    """A frame stamp as an aware datetime: NIS stamps frames in UTC (`nd2_metadata`)."""
    return stamp.astype("datetime64[us]").item().replace(tzinfo=datetime.UTC)


def series_optikit(
    file: File,
    acquisition: NikonAcquisition,
    position: Position,
    calibration: Calibration,
    centre: StageCentre | None,
    bottom_um: float | None,
    data: xr.DataArray,
    descending_z: bool,
    scale_note: str | None,
) -> OptikitStateInput | None:
    """The hardware state of the whole image: the instrument's devices
    (`nd2_instruments`), the run itself, and the stage."""
    focus = f"{position.focus_um:g} um" if position.focus_um is not None else None
    stage = None
    if centre is not None:
        stage = StageStateInput(
            x=f"{centre.stage_x:g} um", y=f"{centre.stage_y:g} um", z=focus
        )
    elif focus is not None:
        # The focus is recorded even where the stage is not: state what is known, and
        # leave x and y unstated rather than at 0.
        stage = StageStateInput(z=focus)

    state_devices = devices(
        *acquisition.devices(),
        # The run itself. The only home the source file's identity has, so `source-file`
        # and `source-file-id` are load-bearing rather than decorative.
        device(
            "acquisition",
            "software",
            setting("software", string=acquisition.scope.software or "NIS-Elements"),
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting(
                "position-index",
                number=float(position.index) if position.index is not None else None,
            ),
            setting("position-name", string=position.name),
            setting("pixel-type", string=str(data.dtype)),
            *(
                setting(f"size-{axis}", number=float(size))
                for axis, size in data.sizes.items()
            ),
            setting("spacing-x", quantity=f"{calibration.x_um:g} um" if calibration.x_um else None),
            setting("spacing-y", quantity=f"{calibration.y_um:g} um" if calibration.y_um else None),
            setting(
                "spacing-z",
                quantity=f"{calibration.z_um:g} um"
                if calibration.z_um and "z" in data.dims
                else None,
            ),
            setting("z-order", string="reversed from top-down" if descending_z else None),
            setting("stack-bottom", quantity=f"{bottom_um:g} um" if bottom_um is not None else None),
            # Says out loud what the registration had to assume. There is no way to
            # express "unknown" through a BY_DIMENSION edge.
            setting("scale-note", string=scale_note),
        ),
        device(
            "stage",
            "stage",
            setting("orientation-source", string=centre.orientation.source if centre else None),
            setting("world-x", number=centre.x if centre else None),
            setting("world-y", number=centre.y if centre else None),
        ),
    )

    if stage is None and not state_devices:
        return None
    return OptikitStateInput(stage=stage, devices=state_devices)


def channel_label(channel: NikonChannel | None, index: int, implicit_rgb: bool) -> str:
    if implicit_rgb:
        return (
            ("Red", "Green", "Blue", "Alpha")[index] if index < 4 else f"Sample {index}"
        )
    if channel is not None and channel.name:
        return channel.name
    return f"Channel {index}"


def channel_color(
    channel: NikonChannel | None, index: int, implicit_rgb: bool, total: int
) -> tuple[int, int, int, int]:
    """What to draw this channel in.

    The colour NIS displayed it in first — the one of these that is a recorded fact
    rather than a guess (black is NIS's "none"). Then its emission, which at least lands
    in the right part of the spectrum. Then a palette, so that two channels never
    silently collide.
    """
    if implicit_rgb:
        # Index 3 of an RGBA pixel is alpha — a mask, not a colour. White labels it
        # without painting it the same red as channel 0.
        if index >= 3:
            return rgba((255, 255, 255))
        return rgba(RGB_PRIMARIES[index])

    if channel is not None:
        if channel.color is not None and any(channel.color):
            return rgba(channel.color)
        emission = (
            sum(channel.emission_band) / 2 if channel.emission_band else channel.emission_nm
        )
        if emission is not None:
            return color_for_wavelength(emission, COLOR_BY_EMISSION, FAR_RED)

    return fallback_color(index, total)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel, as both the anchors and the layers need it.

    Computed once because the histogram is the expensive part — it is the render
    metadata the anchor carries, and computing it twice would double the only forced
    materialisation in the converter.
    """

    index: int
    label: str
    color: tuple[int, int, int, int]
    histogram: ValueHistogramInput
    clim: tuple[float, float]
    nikon: NikonChannel | None


def display_window(shown: np.ndarray) -> tuple[float, float]:
    """The contrast window a channel's layer opens with: p1/p99 of what it *shows*.

    `shown` is the picture, which is not the voxels wherever the layer projects — see
    `channel_specs`. p1/p99 rather than min/max, because a single hot pixel should not
    decide the display range of the whole channel — except where the percentiles coincide
    (a channel that is over 99 % one value), which would be a window of no width.
    """
    shown = shown[np.isfinite(shown)]
    low, high = (float(v) for v in np.percentile(shown, [1, 99]))
    if high <= low:
        low, high = float(shown.min()), float(shown.max())
    if high <= low:
        high = low + 1.0
    return low, high


def channel_specs(
    data: xr.DataArray,
    pyramid: Sequence[xr.DataArray],
    nikon: Sequence[NikonChannel],
    implicit_rgb: bool,
    is_volume: bool,
) -> list[ChannelSpec]:
    """One spec per channel of the array.

    A contrast window is per channel by nature, which is why each histogram is computed
    from that channel's slice alone rather than from the stack. NIS keeps no display
    window this converter can read, so the window is derived from the picture.
    """
    specs: list[ChannelSpec] = []
    total = data.sizes["c"]

    for index in range(total):
        # Packed samples are three array channels behind one ND2 channel.
        channel = nikon[index] if index < len(nikon) and not implicit_rgb else None

        base = data.isel(c=index)
        slice_ = base
        if base.nbytes > HISTOGRAM_BUDGET_BYTES:
            # `from_array` materialises whatever it is handed, so a multi-gigabyte
            # channel is summarised from the finest pyramid level that fits. Max
            # downsampling keeps only block maxima, so every level shifts the
            # distribution up — hence the finest, not the coarsest.
            slice_ = next(
                (
                    level.isel(c=index)
                    for level in pyramid[1:]
                    if level.isel(c=index).nbytes <= HISTOGRAM_BUDGET_BYTES
                ),
                pyramid[-1].isel(c=index),
            )

        values = np.asarray(slice_)

        # What the layer draws. A z-stack is drawn as a maximum projection, and the
        # maxima along z are a different population from the voxels: in sparse
        # fluorescence the stack's p99 sits barely above background while nearly every
        # projected pixel holds something brighter.
        shown = values
        if is_volume:
            projection = base.max("z")
            if slice_ is base:
                shown = values.max(axis=slice_.dims.index("z"))
            elif projection.nbytes <= HISTOGRAM_BUDGET_BYTES:
                shown = np.asarray(projection)
            else:
                shown = values.max(axis=slice_.dims.index("z"))

        specs.append(
            ChannelSpec(
                index=index,
                label=channel_label(channel, index, implicit_rgb),
                color=channel_color(channel, index, implicit_rgb, total),
                histogram=ValueHistogramInput.from_array(values),
                clim=display_window(shown),
                nikon=channel,
            )
        )

    return specs


def channel_anchors(
    specs: Sequence[ChannelSpec],
    acquisition: NikonAcquisition,
    sample_label: str,
    resolution: Sequence[int] | None = None,
) -> list[CoordinateAnchorInput]:
    """One anchor per channel: its contrast window, its identity, its optics — laid out
    the way the instrument that recorded it is built."""
    return [
        CoordinateAnchorInput(
            axis_anchors=(AxisAnchorInput(axis="c", value=spec.index),),
            value_histogram=spec.histogram,
            label=LabelInput(label=spec.label),
            microscope=(
                acquisition.channel_devices(spec.nikon) if spec.nikon is not None else None
            ),
            light_graph=(
                acquisition.channel_lightpath(spec.nikon, sample_label, resolution)
                if spec.nikon is not None
                else None
            ),
        )
        for spec in specs
    ]


def channel_lens(dataset: Any, index: int) -> Any:
    """The lens that pins channel `index`: what a layer over one channel draws.

    A layer draws its lens and nothing narrower, so a channel is a lens of its own.
    A dataset with no channel axis is one signal already, and its whole lens is it.
    """
    if "c" not in dataset.axis_names:
        return dataset.lens()
    return dataset.lens(c=index)


def author_channel_layers(
    mikro: Mikro,
    dataset: ArrayDataset,
    scene: Any,
    specs: Sequence[ChannelSpec],
    is_volume: bool,
    implicit_rgb: bool,
    order_offset: int = 0,
) -> int:
    """One layer per channel of `dataset`, composited into `scene`.

    Per channel, because one layer over a multi-channel dataset means one recipe standing
    in for every channel: no per-channel colour, no per-channel contrast, nothing to
    toggle. It is also the only place NIS's own channel colour can land.

    The one exception is a packed RGB(A) series, one layer over its samples: those
    "channels" are one photograph, and separating them gives toggles that are only ever
    right in one combination. `implicit_rgb` is trusted only as far as it can be checked —
    this function refuses it for any channel count that is not 3 or 4. That refusal is the
    backstop for the whole rule, kept here rather than left to the server because it fires
    first and demotes instead of failing the upload.

    One lens per channel, beside the dataset's whole one: a layer draws its lens and
    nothing narrower, so the channel a layer shows is the one its lens pins.

    **Which kind of layer** — each recipe has a server kind that states it as fields:

    * one channel, flat → ``create_intensity_layer``
    * one channel, through a z-stack → ``create_volume_layer``
    * a packed RGB(A) pixel, flat → ``create_rgb_layer``

    ``create_layer`` and a render graph are left for the one shape none of them can say,
    a packed RGB(A) **z-stack**: an RGB layer carries no projection mode.

    Returns the number of layers created — not ``len(specs)`` in the RGB case.
    """
    lens = dataset.lens()

    if implicit_rgb and len(specs) not in (3, 4):
        logger.warning(
            "Refusing to composite %s channel(s) into one RGB layer; a render graph "
            "covers exactly one channel unless the channels are the 3 or 4 samples of "
            "one pixel. Authoring one layer per channel instead.",
            len(specs),
        )
        implicit_rgb = False

    if implicit_rgb:
        if is_volume:
            # The one shape the flat kinds cannot say. Three primaries even for RGBA:
            # the fourth sample is alpha, a mask rather than a colour. `rgb_graph` says
            # nearly this, but its bare triples are what the server rejects.
            mikro.create_layer(
                lens=lens.id,
                scene=scene.id,
                render_graph=composite_graph(
                    [
                        {
                            "lens": channel_lens(dataset, index).id,
                            "color": rgba(primary),
                            "colormap": ColorMap.INTENSITY,
                            # Already display-referred; stretching it would change the
                            # colours the image was authored with.
                            "clim_min": 0.0,
                            "clim_max": 255.0,
                            "mode": ProjectionMode.MIP,
                        }
                        for index, primary in enumerate(RGB_PRIMARIES)
                    ]
                ),
                blending=Blending.ADDITIVE,
                visible=True,
                order=order_offset,
            )
            return 1

        # Flat: the photograph is exactly what an RGB layer is for, and it says so as
        # three component lenses, one per sample, rather than as a tree a reader has to
        # recognise. One contrast window across all three — they are components of one
        # picture, not three signals with their own scaling — left at the full 8-bit
        # range for the reason above.
        #
        # No `blending` argument here, and none accepted: the server forces NORMAL,
        # because a photograph sits *over* what is beneath it rather than summing with
        # it. That is a change from the ADDITIVE this path used to pass.
        mikro.create_rgb_layer(
            lens=lens.id,
            scene=scene.id,
            red_lens=channel_lens(dataset, 0).id,
            green_lens=channel_lens(dataset, 1).id,
            blue_lens=channel_lens(dataset, 2).id,
            clim_min=0.0,
            clim_max=255.0,
            visible=True,
            order=order_offset,
        )
        return 1

    for spec in specs:
        # `color`, not a colormap: what this converter knows about a channel is an RGBA
        # quad — NIS's channel colour, or the one its emission implies —
        # and no named map means it. That fact used to force the whole layer onto a
        # render graph; an intensity layer carries it as a field now.
        settings = dict(
            lens=channel_lens(dataset, spec.index).id,
            scene=scene.id,
            color=spec.color,
            # Of what the layer shows, not of the voxels — see `channel_specs`.
            clim_min=spec.clim[0],
            clim_max=spec.clim[1],
            blending=Blending.ADDITIVE,
            visible=True,
            order=order_offset + spec.index,
        )

        if is_volume:
            # A z-stack has no single plane to show, so each channel projects through the
            # volume rather than picking one slice. Same kind of layer, one field more.
            mikro.create_volume_layer(mode=ProjectionMode.MIP, **settings)
        else:
            mikro.create_intensity_layer(**settings)

    return len(specs)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preview:
    """One series drawn the way its layers draw it, small: a maximum projection.

    Rendered in `convert_point`, while the pyramid it is read from still exists, and only
    blended into a scene snapshot once the scene's other series have landed.
    `origin_um` and `pixel_um` are ``(y, x)`` in the world, or None for a series that is
    not registered into it — such a series is the whole of its own scene.
    """

    rgb: np.ndarray
    origin_um: tuple[float, float] | None
    pixel_um: tuple[float, float] | None


def preview_level(pyramid: Sequence[xr.DataArray], side: int) -> int:
    """The finest pyramid level whose longer side fits `side`, else the coarsest."""
    for index, level in enumerate(pyramid):
        if max(level.sizes.get("y", 1), level.sizes.get("x", 1)) <= side:
            return index
    return len(pyramid) - 1


def render_preview(
    level: xr.DataArray, specs: Sequence[ChannelSpec], implicit_rgb: bool, side: int
) -> np.ndarray:
    """What the layers over `level` show, as 8-bit RGB: each channel's maximum through z,
    in its own colour and contrast window, summed — the scene blends ADDITIVE.

    The first time point and the first index of any other axis: a snapshot is one
    picture. An RGBA series' alpha is a mask, and is not drawn, as its layer does not.
    """
    level = level.isel({d: 0 for d in level.dims if d not in ("c", "z", "y", "x")})
    if "z" in level.dims:
        level = level.max("z")
    step = -(-max(level.sizes["y"], level.sizes["x"]) // side)
    level = level.isel(y=slice(None, None, step), x=slice(None, None, step))

    rgb = np.zeros((level.sizes["y"], level.sizes["x"], 3), dtype=np.float32)
    for spec in specs:
        if implicit_rgb and spec.index >= 3:
            continue
        low, high = spec.clim
        values = np.asarray(level.isel(c=spec.index), dtype=np.float32)
        shown = np.clip((values - low) / (high - low), 0.0, 1.0)
        rgb += shown[..., None] * np.asarray(spec.color[:3], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mosaic(previews: Sequence[Preview]) -> np.ndarray:
    """The previews of one scene, each at its place in the world.

    One preview is its own picture. Several are laid out at their stage positions on one
    canvas at the finest pixel size among them (coarser where that would exceed
    `SNAPSHOT_PIXELS`), each resampled to it nearest-neighbour;
    where tiles overlap the brighter wins, rather than the overlap summing to a seam.
    """
    if len(previews) == 1 or any(p.origin_um is None for p in previews):
        return previews[0].rgb

    pixel = min(min(p.pixel_um) for p in previews)
    top = min(p.origin_um[0] for p in previews)
    left = min(p.origin_um[1] for p in previews)
    bottom = max(p.origin_um[0] + p.rgb.shape[0] * p.pixel_um[0] for p in previews)
    right = max(p.origin_um[1] + p.rgb.shape[1] * p.pixel_um[1] for p in previews)
    # Coarser where the tiles span more than the snapshot's side: a strip of tiles, or
    # two fields far apart, would otherwise grow the canvas with the distance between.
    pixel = max(pixel, max(bottom - top, right - left) / SNAPSHOT_PIXELS)

    shape = (max(1, round((bottom - top) / pixel)), max(1, round((right - left) / pixel)))
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for preview in previews:
        rows = max(1, round(preview.rgb.shape[0] * preview.pixel_um[0] / pixel))
        cols = max(1, round(preview.rgb.shape[1] * preview.pixel_um[1] / pixel))
        y0 = min(round((preview.origin_um[0] - top) / pixel), shape[0] - 1)
        x0 = min(round((preview.origin_um[1] - left) / pixel), shape[1] - 1)
        rows, cols = min(rows, shape[0] - y0), min(cols, shape[1] - x0)
        picked = preview.rgb[
            (np.arange(rows) * preview.rgb.shape[0] // rows)[:, None],
            np.arange(cols) * preview.rgb.shape[1] // cols,
        ]
        target = canvas[y0 : y0 + rows, x0 : x0 + cols]
        np.maximum(target, picked, out=target)
    return canvas


def encode_png(rgb: np.ndarray) -> bytes:
    """`rgb` (8-bit, height × width × 3) as a PNG, with nothing but the standard library."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body))
        )

    height, width, _ = rgb.shape
    # Every scanline is prefixed with its filter type; 0 is none.
    rows = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)], axis=1
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(np.ascontiguousarray(rows).tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def snapshot_scene(
    mikro: Mikro, scene: Scene, entries: Sequence[Converted], scratch: str
) -> None:
    """Attach a maximum projection of `scene` as its snapshot: a preview a client can
    show without compositing the layers itself."""
    previews = [entry.preview for entry in entries if entry.preview is not None]
    if not previews:
        return
    path = os.path.join(scratch, f"snapshot-{scene.id}.png")
    with open(path, "wb") as out:
        out.write(encode_png(mosaic(previews)))
    with open(path, "rb") as picture:
        mikro.create_scene_snapshot(
            file=picture, scene=scene.id, name=f"{scene.name} (maximum projection)"
        )


def compose_scene(
    mikro: Mikro, name: str, coordinate_system: Any, entries: Sequence[Converted]
) -> Scene:
    """A scene over `coordinate_system`, with every layer authored explicitly.

    ``create_scene`` rather than ``stage`` / ``create_scene_from_coordinate_system``:
    the bootstrap decides on your behalf how many registered sources become layers,
    which recipe each one gets, and — for anything multi-channel — collapses a whole
    dataset into a single layer. This converter knows exactly what it uploaded and what
    each channel is, so none of those are decisions worth delegating. It adopts the
    world system, then authors one lens per dataset and one layer per channel, each
    with its own colour, contrast window and projection.

    `entries` is one series, or the tiles of one mosaic — never the whole file; which
    series share a scene is the action's decision (see `share_a_scene`), not this
    function's.

    The scene carries only what is genuinely scene-wide. ``blending`` is ADDITIVE
    because that is how fluorescence channels combine — and how three RGB primaries
    recompose a photograph. ``preferred_view`` is a viewer hint and nothing more: the
    projection that actually renders a volume lives on each layer's channel node, so
    this only says which way a viewer should open, and only claims a preference when
    something in the scene has depth to look into.
    """
    scene = mikro.create_scene(
        name=name,
        coordinate_system=coordinate_system.id,
        blending=Blending.ADDITIVE,
        preferred_view=(
            PreferredView.THREE_D
            if any(entry.is_volume for entry in entries)
            else PreferredView.TWO_D
        ),
    )

    # Ordered so a series' channels stay together in the layer stack rather than
    # interleaving with the next series'.
    order = 0
    for entry in entries:
        order += author_channel_layers(
            mikro,
            entry.dataset,
            scene,
            entry.specs,
            entry.is_volume,
            entry.implicit_rgb,
            order,
        )

    # Nominated through the whole lens: on an unsliced lens that is the dataset's own
    # default scene, and so where its thumbnail (`latestSnapshot`) comes from. A
    # mosaic's tiles all nominate the mosaic.
    for entry in entries:
        mikro.set_lens_default_scene(lens=entry.dataset.lens().id, scene=scene.id)

    logger.info(
        "Scene %s over %s: %s layer(s) from %s dataset(s)",
        scene.id,
        scene.world_coordinate_system.name,
        order,
        len(entries),
    )
    return scene


def stage_group(
    mikro: Mikro,
    title: str,
    world: Any,
    entries: Sequence[Converted],
    scratch: str | None = None,
) -> None:
    """Compose the scene(s) for one group out of `scene_groups`.

    The series placed in the world share one scene over it. Any other series is staged
    alone: over its own micrometer space when it is calibrated but its position is not
    fully known (`world_offsets`), over its own pixel grid when it is not calibrated
    either. Both are the honest frame for it, and one it shares with nothing.

    With a `scratch` directory, every scene also gets its snapshot (`snapshot_scene`).
    """

    def compose(name: str, system: Any, members: Sequence[Converted]) -> None:
        scene = compose_scene(mikro, name, system, members)
        if scratch is None:
            return
        try:
            snapshot_scene(mikro, scene, members, scratch)
        except Exception:  # the scene is composed; a picture of it is not worth it
            logger.exception("Could not snapshot %s", name)

    registered = [entry for entry in entries if entry.registered]
    if registered:
        layer_count = sum(entry.layer_count for entry in registered)
        if layer_count > MAX_SCENE_LAYERS:
            logger.warning(
                "%s would compose %s layers from %s series; not building a scene over "
                "more than %s. The world holds every registration and outlives this run, "
                "so a subset can be composed later.",
                title,
                layer_count,
                len(registered),
                MAX_SCENE_LAYERS,
            )
        else:
            compose(title, world, registered)

    for entry in entries:
        if entry.registered:
            continue
        if entry.own_space is not None:
            compose(entry.name, entry.own_space, [entry])
            continue
        if entry.dataset.intrinsic_system is None:
            logger.warning(
                "Series %s has no intrinsic coordinate system to compose over; the "
                "dataset is uploaded but has no scene.",
                entry.series,
            )
            continue
        compose(entry.name, entry.dataset.intrinsic_system, [entry])



def share_a_scene(points: Sequence[Sequence[Converted]]) -> bool:
    """Whether these points' fields touch, and so make one picture: a grid NIS acquired
    but did not stitch. Each point is the datasets made of it (several for a colour
    camera with several channels), and its field is theirs together. The wells of a plate, or points picked across a slide, do not —
    a scene over them is mostly empty space, and past `MAX_SCENE_LAYERS` it is none at
    all — so each of those gets a scene of its own.

    Touching means connected: every field reaches the others through fields that
    overlap it or sit within `MOSAIC_GAP_FRACTION` of a field of it.
    """
    return touching([_union([entry.footprint for entry in point]) for point in points])


def touching(boxes: Sequence[tuple[float, float, float, float] | None]) -> bool:
    """Whether these ``(y0, x0, y1, x1)`` fields are all connected through fields that
    touch; never for fewer than two, or for one that is not placed."""
    if len(boxes) < 2 or any(box is None for box in boxes):
        return False
    reached = {0}
    frontier = [0]
    while frontier:
        i = frontier.pop()
        for j, other in enumerate(boxes):
            if j not in reached and _touch(boxes[i], other):
                reached.add(j)
                frontier.append(j)
    return len(reached) == len(boxes)


def _union(boxes):
    if not boxes or any(box is None for box in boxes):
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _touch(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Whether two ``(y0, x0, y1, x1)`` boxes overlap or nearly abut."""
    gap_y = MOSAIC_GAP_FRACTION * min(a[2] - a[0], b[2] - b[0])
    gap_x = MOSAIC_GAP_FRACTION * min(a[3] - a[1], b[3] - b[1])
    return (
        a[0] <= b[2] + gap_y
        and b[0] <= a[2] + gap_y
        and a[1] <= b[3] + gap_x
        and b[1] <= a[3] + gap_x
    )


# ---------------------------------------------------------------------------
# One point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """What the action needs to know about a series after it has landed."""

    series: int
    name: str
    dataset: ArrayDataset
    specs: list[ChannelSpec]
    #: Placed in the file's shared world (`world_offsets`).
    registered: bool
    is_volume: bool
    implicit_rgb: bool
    preview: Preview | None = None
    #: The micrometer space of its own a calibrated series that is not placed in the
    #: world is registered into; None for the placed and the uncalibrated.
    own_space: Any | None = None
    #: Its field in the world, ``(y0, x0, y1, x1)`` in micrometers, when it is placed.
    footprint: tuple[float, float, float, float] | None = None
    #: The name of the point it was made of: its scene's, unless it shares a mosaic.
    point_name: str = ""

    @property
    def layer_count(self) -> int:
        """How many layers this series composes into a scene.

        Not ``len(specs)``: a packed RGB(A) series is one layer over its 3 or 4
        samples. Kept here rather than recomputed at each call site so the budget check
        in `stage_group` and what `author_channel_layers` actually creates cannot drift.

        For every other series this is ``len(specs)`` — one layer per channel — and that
        is the invariant, not an implementation detail.
        """
        return 1 if self.implicit_rgb else len(self.specs)


def packs_one_pixel(packed_samples: bool, channels: int, dtype: Any) -> bool:
    """Whether this image's channels are the samples of one pixel — the ONLY exception.

    ONE LAYER PER CHANNEL. Exactly one thing may say yes: nd2's sample axis
    (``packed_samples`` out of `field_array`), which it exposes only for a colour
    camera's pixel. Nothing that *counts* channels may: collapsing three dyes destroys
    per-channel colour, contrast and toggling.

    And the samples have to be 8-bit. An RGB layer is drawn unstretched over 0..255; a
    deeper sample axis gets a layer and a window per sample instead.
    """
    return packed_samples and channels in (3, 4) and np.dtype(dtype) == np.uint8


@dataclass(frozen=True)
class FileContext:
    """What every point of one file shares: read once, before the first upload."""

    f: Any
    file: File
    acquisition: NikonAcquisition
    channels: list[NikonChannel]
    calibration: Calibration
    stack: ZStack | None
    frames: FrameTable
    whole: xr.DataArray
    world: Any
    levels: int
    method: str

    @property
    def descending_z(self) -> bool:
        return self.stack is not None and not self.stack.bottom_to_top


def convert_point(
    mikro: Mikro,
    context: FileContext,
    position: Position,
    rgb_channel: int | None,
    scratch: str,
    preview_side: int | None = None,
) -> Converted:
    """Upload one point (one channel of it, for a multi-channel colour camera), and place
    it if the file says where. Never stages it: that depends on the points around it —
    see `share_a_scene`.

    Placed means in the shared world, and only when the file says where along every
    spatial axis (`world_offsets`). A calibrated image it does not place is registered
    into a micrometer space of its own, at its origin: its scale holds, its position is
    claimed against nothing.
    """
    file, acquisition = context.file, context.acquisition
    array, extra_types, packed_samples = field_array(
        context.whole, position.index, rgb_channel, context.descending_z
    )

    # Canonical order — time, then channel, then space — is load-bearing: the render
    # axes are derived from the *position* of the spatial axes.
    data = canonical(array, types=extra_types)
    dims = [str(d) for d in data.dims]

    implicit_rgb = packs_one_pixel(packed_samples, data.sizes["c"], data.dtype)
    if packed_samples and not implicit_rgb:
        logger.warning(
            "%s has a packed sample axis but %s %s sample(s) are no 8-bit RGB(A) "
            "picture; treating them as channels, one layer each.",
            file.name,
            data.sizes["c"],
            data.dtype,
        )
    is_volume = data.sizes.get("z", 1) > 1

    pyramid = image_pyramid(
        data, context.levels or auto_levels(data.sizes), context.method, extra_types, scratch
    )
    scales = scales_from(pyramid, context.method)

    scale, scale_note = voxel_scale(context.calibration, dims)
    centre = stage_centre(position, orientation_of(acquisition.scope))
    bottom = stack_bottom(context.frames, position.index, context.stack) if is_volume else None
    whole_window, timepoint_windows = frame_windows(
        context.frames, position.index, data.sizes.get("t", 1)
    )

    channels = context.channels
    if rgb_channel is not None:
        channels = [channels[rgb_channel]] if rgb_channel < len(channels) else []
    specs = channel_specs(data, pyramid, channels, implicit_rgb, is_volume)

    name = file.name
    if position.index is not None:
        name += f" - {position.name or f'point {position.index + 1}'}"
    point_name = name
    if rgb_channel is not None:
        label = channels[0].name if channels and channels[0].name else f"channel {rgb_channel}"
        name += f" - {label}"

    series_identifier = "image"
    if position.index is not None:
        series_identifier = f"P{position.index}"
    if rgb_channel is not None:
        series_identifier += f":C{rgb_channel}"

    image_anchor = CoordinateAnchorInput(
        axis_anchors=(),
        acquisition_window=whole_window,
        ome_metadata=OmeMetadataInput(
            metadata_string=image_record(
                context.f, file.name, str(file.id), position, rgb_channel
            )
        ),
        microscope=series_optikit(
            file,
            acquisition,
            position,
            context.calibration,
            centre,
            bottom,
            data,
            context.descending_z,
            scale_note,
        ),
    )

    dataset = mikro.create_array_dataset(
        data=pyramid[0],
        scales=scales,
        name=name,
        axes=axes_for(
            data, types=dict.fromkeys(extra_types, "INDEX"), long_names=AXIS_LONG_NAMES
        ),
        anchors=[
            # No label on the image anchor: a label names a channel, and the image's name
            # is the dataset's.
            image_anchor,
            *channel_anchors(
                specs,
                acquisition,
                position.name or "Sample",
                (data.sizes["x"], data.sizes["y"]),
            ),
            # When each timepoint was taken: intervals a nominal period cannot state.
            *(
                CoordinateAnchorInput(
                    axis_anchors=(AxisAnchorInput(axis="t", value=index),),
                    acquisition_window=window,
                )
                for index, window in timepoint_windows
            ),
        ],
        # The link back to the bytes these arrays were read out of: the point (and the
        # channel) of the file — what survives a reader that lists them in another order.
        source_files=[
            SourceFileInput(
                file=file.id,
                series_identifier=series_identifier,
                # nd2 hands back the stored values and this converter only renames and
                # reorders axes (and turns a top-down stack over): the histograms are
                # true of the file too.
                value_relation=ValueRelation.IDENTICAL,
            )
        ],
    )

    offsets = world_offsets(centre, scale, data, bottom) if scale is not None else None
    footprint = None
    if scale is not None and offsets is not None:
        footprint = (
            offsets["y"],
            offsets["x"],
            offsets["y"] + data.sizes["y"] * scale["y"],
            offsets["x"] + data.sizes["x"] * scale["x"],
        )

    preview = None
    if preview_side is not None:
        try:
            level = pyramid[preview_level(pyramid, preview_side)]
            rgb = render_preview(level, specs, implicit_rgb, preview_side)
            placed = footprint is not None
            preview = Preview(
                rgb,
                (offsets["y"], offsets["x"]) if placed else None,
                (
                    scale["y"] * data.sizes["y"] / rgb.shape[0],
                    scale["x"] * data.sizes["x"] / rgb.shape[1],
                )
                if placed
                else None,
            )
        except Exception:  # only costs the snapshot
            logger.exception("Could not render a preview of %s", name)

    series = position.index or 0
    if scale is None:
        logger.info("%s: %s", name, scale_note)
        return Converted(
            series,
            name,
            dataset,
            specs,
            False,
            is_volume,
            implicit_rgb,
            preview,
            point_name=point_name,
        )

    if offsets is None:
        logger.info(
            "%s: the file does not say where it sits along every axis; registered into a "
            "space of its own rather than the shared world",
            name,
        )
        own = space_3d(mikro, f"{name} (micrometers)", unit="micrometer")
        own.register(dataset, validity=PlacementValidity.INFERRED, scale=scale)
        return Converted(
            series,
            name,
            dataset,
            specs,
            False,
            is_volume,
            implicit_rgb,
            preview,
            own,
            point_name=point_name,
        )

    context.world.register(
        dataset, validity=PlacementValidity.INFERRED, scale=scale, **offsets
    )
    return Converted(
        series,
        name,
        dataset,
        specs,
        True,
        is_volume,
        implicit_rgb,
        preview,
        footprint=footprint,
        point_name=point_name,
    )


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


def units_of(
    f: Any, positions: Sequence[Position]
) -> list[tuple[Position, int | None]]:
    """What becomes a dataset: every point, and — for a colour camera with several
    channels — every channel of it."""
    rgb_channels: list[int | None] = [None]
    if "S" in f.sizes and "C" in f.sizes:
        rgb_channels = list(range(int(f.sizes["C"])))
    return [(position, c) for position in positions for c in rgb_channels]


def layers_per_point(f: Any) -> int:
    """How many layers one point will compose into: ``C`` either way — one per channel,
    or, with a sample axis, one RGB layer per colour picture."""
    return int(f.sizes.get("C", 1))


def planned_mosaic(
    f: Any, positions: Sequence[Position], scope: NikonScope, calibration: Calibration
) -> bool:
    """Whether the points will share one mosaic scene, decided from the header before any
    pixel is read, so that each snapshot is rendered at the size its scene will have. The
    same rule `stage_points` applies to what actually landed."""
    if len(positions) < 2 or calibration.x_um is None or calibration.y_um is None:
        return False
    if len(positions) * layers_per_point(f) > MAX_SCENE_LAYERS:
        return False
    orientation = orientation_of(scope)
    height = (int(f.sizes["Y"]) - 1) * calibration.y_um
    width = (int(f.sizes["X"]) - 1) * calibration.x_um
    boxes = []
    for position in positions:
        centre = stage_centre(position, orientation)
        boxes.append(
            None
            if centre is None
            else (
                centre.y - height / 2,
                centre.x - width / 2,
                centre.y + height / 2,
                centre.x + width / 2,
            )
        )
    return touching(boxes)


def stage_points(
    mikro: Mikro,
    file: File,
    world: Any,
    entries: Sequence[Converted],
    scratch: str | None,
) -> None:
    """One mosaic scene for points whose fields touch, a scene each otherwise. The
    datasets of one point (the channels of a colour camera) always share its scene.

    A mosaic over more than `MAX_SCENE_LAYERS` layers is not composed: `stage_group`
    would refuse it, and leave a large unstitched grid without any scene. Its points get
    a scene each instead.
    """
    by_point: dict[int, list[Converted]] = {}
    for entry in entries:
        by_point.setdefault(entry.series, []).append(entry)
    points = list(by_point.values())
    layers = sum(entry.layer_count for entry in entries if entry.registered)
    if len(points) > 1 and layers <= MAX_SCENE_LAYERS and share_a_scene(points):
        stage_group(mikro, f"{file.name} ({len(points)} points)", world, entries, scratch)
        return
    for point in points:
        try:
            stage_group(mikro, point[0].point_name, world, point, scratch)
        except Exception:  # one scene is not worth the others
            logger.exception("Could not stage %s", point[0].point_name)


@app.action(
    name="Convert Nikon ND2 file to Datasets",
    description=(
        "Read a Nikon ND2 file natively (no Bio-Formats) and upload every stage position "
        "as a pyramided dataset with NIS-Elements' own metadata: modalities, laser lines, "
        "detection bands, camera, detectors and pinhole, placed in a shared micrometer "
        "space at its stage position where the file records it along every axis "
        "(otherwise in a calibrated space of its own), and staged in a scene — one mosaic "
        "for points whose fields touch — with a maximum projection as its snapshot."
    ),
    collections=["conversion"],
    logo=logo_of("convert_nd2_file_to_datasets"),
)
def convert_nd2_file_to_datasets(
    file: File,
    mikro: Mikro,
    task: Task,
    levels: int = 0,
    method: str = "max",
    stage_scene: bool = True,
    world_name: str = "",
    snapshot: bool = True,
) -> list[ArrayDataset]:
    """Convert every stage position of an ND2 file into a dataset.

    `levels` of 0 picks the pyramid depth per image (`auto_levels`). Returns a list
    rather than yielding, for the reason the OME converter gives: a mosaic scene is
    composed after its last point, and an abandoned generator would leave it unstaged.
    """
    assert file.store, "No file provided"
    if not file.name.lower().endswith(READABLE_SUFFIXES):
        raise ValueError(f"{file.name} is not an ND2 file.")

    task.progress(0, "Downloading file")
    # Under its own name, into a scratch dir of its own — never the working directory,
    # which the `finally` below would then delete from. MIKRO_SCRATCH_DIR when set,
    # TMPDIR otherwise; a tmpfs /tmp puts the download in RAM.
    workdir = tempfile.mkdtemp(
        prefix="nd2-converter-", dir=os.environ.get("MIKRO_SCRATCH_DIR") or None
    )

    try:
        path = file.store.download(os.path.join(workdir, os.path.basename(file.name)))
        task.progress(5, "Reading the ND2 header")

        import nd2

        with nd2.ND2File(path) as f:
            if f.is_legacy:
                raise ValueError(
                    f"{file.name} is a legacy (JPEG 2000) ND2 file, which this converter "
                    "does not read; re-save it from a current NIS-Elements."
                )

            tree = picture_metadata(f)
            scope = read_scope(f, tree)
            acquisition = acquisition_of(scope)
            logger.info(
                "%s was recorded on %s (model %s, modalities %s)",
                file.name,
                scope.microscope or "an unnamed stand",
                acquisition.instrument.key,
                ", ".join(scope.modalities) or "none",
            )

            # One per invocation: a physical space belongs to nobody and outlives every
            # dataset put into it. Two invocations mint two disjoint worlds.
            world = space_3d(
                mikro, world_name or f"{file.name} (micrometers)", unit="micrometer"
            )
            frames = read_frame_table(f)
            positions = read_positions(f, frames)
            context = FileContext(
                f=f,
                file=file,
                acquisition=acquisition,
                channels=read_channels(f, tree),
                calibration=read_calibration(f),
                stack=z_stack(f),
                frames=frames,
                whole=file_array(f),
                world=world,
                levels=levels,
                method=method,
            )

            units = units_of(f, positions)
            preview_side = None
            if snapshot and stage_scene:
                # A mosaic's points share the snapshot's side; a point staged alone has
                # all of it.
                shared = planned_mosaic(f, positions, scope, context.calibration)
                preview_side = (
                    max(64, int(SNAPSHOT_PIXELS / np.sqrt(len(positions))))
                    if shared
                    else SNAPSHOT_PIXELS
                )

            entries: list[Converted] = []
            failures: list[tuple[str, Exception]] = []
            for done, (position, rgb_channel) in enumerate(units):
                where = (
                    f"point {position.index + 1}/{len({p.index for p, _ in units})}"
                    if position.index is not None
                    else "the image"
                ) + (f", channel {rgb_channel}" if rgb_channel is not None else "")
                task.progress(10 + int(85 * done / len(units)), f"Converting {where}")
                # Per point, and gone once it is uploaded: 96 wells' persisted levels
                # would otherwise all wait for the end of the file.
                scratch = tempfile.mkdtemp(prefix="point-", dir=workdir)
                try:
                    entries.append(
                        convert_point(
                            mikro, context, position, rgb_channel, scratch, preview_side
                        )
                    )
                except Exception as exc:  # one bad point must not cost the other 95
                    logger.exception("%s of %s failed", where, file.name)
                    failures.append((where, exc))
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)

            datasets = [entry.dataset for entry in entries]

            if stage_scene and entries:
                try:
                    stage_points(mikro, file, world, entries, workdir if snapshot else None)
                except Exception:  # the datasets are uploaded; a scene is not worth them
                    logger.exception("Could not stage %s", file.name)

        if failures and not datasets:
            raise RuntimeError(
                f"All {len(failures)} point(s) failed; first: {failures[0][1]}"
            ) from failures[0][1]
        if failures:
            logger.warning(
                "%s point(s) failed: %s", len(failures), [where for where, _ in failures]
            )

        task.progress(100, f"Converted {len(datasets)} dataset(s)")
        return datasets

    finally:
        # The whole scratch directory, and only ever that directory, which this
        # converter made and owns.
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(workdir):
            # Cleanup must never mask whatever actually went wrong above.
            logger.warning("Could not remove the scratch directory %s", workdir)


if __name__ == "__main__":
    run(app)
