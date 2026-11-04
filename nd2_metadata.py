"""What NIS-Elements recorded about an ND2 acquisition, read into plain dataclasses.

Two sources, kept apart:

- ``nd2``'s **typed** API — ``metadata`` (per channel: name, colour, emission and
  excitation wavelength, objective, pinhole, modality flags, calibration and the
  camera's orientation), ``experiment`` (the T/P/Z loops), ``frame_metadata`` (where the
  stage was and when, per frame), ``events`` (the "Recorded Data" table: per-frame time
  and focus coordinates). This is what the converter relies on.
- ``unstructured_metadata()`` — NIS's own ``SLxPictureMetadata`` tree. nd2 calls it
  experimental, so everything read from it is best-effort and degrades to absent: the
  camera's name, exposure and binning, the objective's order code, the filter path (the
  laser lines and the detection band), and a point scanner's PMT voltages and laser
  powers. Each picture plane names the sample setting it was taken with
  (``PlaneNew/<i>/SampleIndex``); a widefield file has one setting per channel, an A1 one
  for all.

Facts NIS did not record stay ``None``. In particular a stage position of exactly
(0, 0) is what a system without an encoded stage (the Ti2 simulator, for one) writes,
not a position (`stage_xy`).
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import logging
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from mikro.api.schema import ObjectiveImmersion

logger = logging.getLogger(__name__)

# The Julian day number of the Unix epoch. NIS stamps every frame with an absolute JDN,
# in UTC: ``top_to_bottom.nd2`` is stamped 16:00:19Z where its text info, in CEST, says
# 18:00:17 (two seconds earlier: the text is when the run started, the stamp the frame).
UNIX_EPOCH_JDN = 2440587.5

# Modality flags that mean the light came through the sample rather than from it.
TRANSMITTED = frozenset({"brightfield", "diContrast", "phaseContrast", "transmitDetector"})

# Filter names NIS writes when there is no named filter.
UNNAMED_FILTERS = frozenset({"", "???", "custom", "<<blank>>"})

# Objective names name their immersion: "Plan Apo 100x Oil DIC H", "Plan Apo Lambda S 40XC Sil",
# "CFI Apo LWD 40x WI". Silicone oil has no member of its own.
IMMERSION_TOKENS = (
    (re.compile(r"\boil\b", re.IGNORECASE), ObjectiveImmersion.OIL),
    (re.compile(r"\bsil\b|silicone", re.IGNORECASE), ObjectiveImmersion.OTHER),
    (re.compile(r"\bglyc", re.IGNORECASE), ObjectiveImmersion.GLYCEROL),
    (re.compile(r"\b(wi|w|water)\b", re.IGNORECASE), ObjectiveImmersion.WATER),
)


def nm(nanometers: float | None) -> str | None:
    """A wavelength in nanometres, as a Length scalar string."""
    if nanometers is None:
        return None
    return f"{nanometers:g} nm"


def um(micrometers: float | None) -> str | None:
    """A length in micrometers, as a Length scalar string."""
    if micrometers is None:
        return None
    return f"{micrometers:g} um"


def _number(value: Any) -> float | None:
    """A finite float out of a metadata value, or None (NIS writes 0 for "not set" often,
    but not always — callers decide whether 0 means absent)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _get(tree: Any, *path: str) -> Any:
    """``tree[path[0]][path[1]]…``, or None where any step is missing."""
    for key in path:
        if not isinstance(tree, Mapping) or key not in tree:
            return None
        tree = tree[key]
    return tree


def _items(tree: Any) -> list[Any]:
    """The children of an NIS array node (``{"0": …, "1": …}``), in index order."""
    if isinstance(tree, Mapping):
        def order(key: str) -> tuple[int, str]:
            digits = re.sub(r"\D", "", str(key))
            return (int(digits) if digits else 0, str(key))

        return [tree[key] for key in sorted(tree, key=order)]
    if isinstance(tree, list):
        return list(tree)
    return []


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NikonCamera:
    """One sample setting's camera (or, on a point scanner, its scan head)."""

    name: str | None = None
    family: str | None = None
    exposure_ms: float | None = None
    binning: str | None = None
    format: str | None = None


@dataclass(frozen=True)
class NikonDetector:
    """A point scanner's detector for one channel: the A1/AX ``CH<n>`` settings."""

    name: str
    high_voltage: float | None = None
    offset: float | None = None
    laser_power: float | None = None


@dataclass(frozen=True)
class NikonChannel:
    """One channel, as NIS recorded it."""

    index: int
    name: str | None
    color: tuple[int, int, int] | None
    emission_nm: float | None
    excitation_nm: float | None
    modalities: tuple[str, ...] = ()
    pinhole_um: float | None = None
    bits: int | None = None
    filter_name: str | None = None
    #: Every line the filter path lists as excitation — the lasers that fired, on a
    #: laser system.
    excitation_lines: tuple[float, ...] = ()
    #: The detection band, where NIS recorded its width (``Wavelength ± TValue``).
    emission_band: tuple[float, float] | None = None
    camera: NikonCamera | None = None
    detector: NikonDetector | None = None

    @property
    def transmitted(self) -> bool:
        """Whether this channel is light through the sample: brightfield, DIC, phase."""
        return bool(TRANSMITTED & set(self.modalities)) and "fluorescence" not in self.modalities


@dataclass(frozen=True)
class NikonScope:
    """What one file says about the instrument as a whole."""

    microscope: str | None = None
    objective_name: str | None = None
    objective_code: str | None = None
    magnification: float | None = None
    numerical_aperture: float | None = None
    refractive_index: float | None = None
    zoom: float | None = None
    projective_magnification: float | None = None
    #: The camera's image axes against the stage's, as NIS stores it: ``(a, b, c, d)``.
    camera_matrix: tuple[float, ...] | None = None
    #: Pixel → stage, where NIS calibrated it: ``(m00, m01, tx, m10, m11, ty)``.
    pixel_to_stage: tuple[float, ...] | None = None
    software: str | None = None
    #: The union of every channel's modality flags.
    modalities: tuple[str, ...] = ()
    camera: NikonCamera | None = None
    z_device: str | None = None

    @property
    def immersion(self) -> ObjectiveImmersion | None:
        """The objective's immersion: from its name, else from the refractive index."""
        if self.objective_name:
            for pattern, kind in IMMERSION_TOKENS:
                if pattern.search(self.objective_name):
                    return kind
        index = self.refractive_index
        if index is None:
            return None
        if index < 1.1:
            return ObjectiveImmersion.AIR
        if index < 1.37:
            return ObjectiveImmersion.WATER
        if index < 1.45:
            return ObjectiveImmersion.OTHER
        if index < 1.49:
            return ObjectiveImmersion.GLYCEROL
        return ObjectiveImmersion.OIL

    @property
    def camera_rotation_deg(self) -> float | None:
        """How far the camera is turned against the stage, in degrees — what a placement
        by stage centre cannot express, and so is recorded instead."""
        if not self.camera_matrix or len(self.camera_matrix) != 4:
            return None
        a, b = self.camera_matrix[0], self.camera_matrix[1]
        return math.degrees(math.atan2(b, a))


@dataclass(frozen=True)
class Calibration:
    """Micrometers per pixel along x, y and z; None along an axis NIS did not calibrate."""

    x_um: float | None
    y_um: float | None
    z_um: float | None


@dataclass(frozen=True)
class Position:
    """One point of a multipoint (XYPosLoop) acquisition, or the single field."""

    index: int | None
    name: str | None
    #: Where the stage put the field's centre, in stage micrometers; None when the stage
    #: was not read out (`stage_xy`).
    x_um: float | None
    y_um: float | None
    #: The focus, as the frame metadata states it. NOT a plane's z: it is constant over a
    #: stack, and sometimes the bottom plane, sometimes the home one.
    focus_um: float | None


@dataclass(frozen=True)
class ZStack:
    """The z loop as recorded."""

    count: int
    step_um: float | None
    bottom_to_top: bool
    home_index: int | None
    device: str | None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def picture_metadata(f: Any) -> Mapping[str, Any]:
    """The ``SLxPictureMetadata`` tree, or empty: best-effort, as nd2 asks."""
    # The chunk is ``…LV|0`` in modern files and ``…|0`` in older ones; asking for one
    # that is not there only warns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key in ("ImageMetadataSeqLV|0", "ImageMetadataSeq|0"):
            try:
                tree = f.unstructured_metadata(include={key})
            except Exception:
                logger.exception("Could not read the unstructured ND2 metadata")
                return {}
            found = _get(tree, key, "SLxPictureMetadata")
            if isinstance(found, Mapping):
                return found
    return {}


def _camera(setting: Any) -> NikonCamera | None:
    camera = _get(setting, "CameraSetting")
    if not isinstance(camera, Mapping):
        return None
    binning_x = _positive(_get(camera, "FormatQuality", "Desc", "BinningX"))
    binning_y = _positive(_get(camera, "FormatQuality", "Desc", "BinningY"))
    result = NikonCamera(
        name=_text(_get(camera, "CameraUserName")),
        family=_text(_get(camera, "CameraFamilyName")),
        exposure_ms=_positive(_get(setting, "ExposureTime"))
        or _positive(_get(camera, "PropertiesQuality", "Exposure")),
        binning=f"{binning_x:g}x{binning_y:g}" if binning_x and binning_y else None,
        format=_text(_get(camera, "FormatQuality", "Desc", "FormatText")),
    )
    return result if any(dataclasses.astuple(result)) else None


def _filter_path(plane: Any) -> tuple[str | None, tuple[float, ...], tuple[float, float] | None]:
    """The name, the excitation lines and the detection band of one plane's filter path."""
    names: list[str] = []
    lines: list[float] = []
    band: tuple[float, float] | None = None
    for item in _items(_get(plane, "FilterPath", "Filter")):
        name = _text(_get(item, "UserName")) or _text(_get(item, "Name"))
        if name and name.lower() not in UNNAMED_FILTERS:
            names.append(name)
        for point in _items(_get(item, "ExcitationSpectrum", "Point")):
            wavelength = _positive(_get(point, "Wavelength"))
            if wavelength is not None and wavelength not in lines:
                lines.append(wavelength)
        for point in _items(_get(item, "EmissionSpectrum", "Point")):
            wavelength = _positive(_get(point, "Wavelength"))
            half = _positive(_get(point, "TValue"))
            if wavelength is not None and half is not None and band is None:
                band = (wavelength - half, wavelength + half)
    return (" / ".join(names) or None), tuple(lines), band


def _scan_detectors(setting: Any) -> dict[str, NikonDetector]:
    """An A1/AX scan head's per-channel detector settings, keyed by the dye it names."""
    quality = _get(setting, "CameraSetting", "PropertiesQuality")
    if not isinstance(quality, Mapping):
        return {}
    found: dict[str, NikonDetector] = {}
    for key, dye in quality.items():
        match = re.fullmatch(r"CH(\d+)ChannelDyeName", str(key))
        if not match or not _text(dye):
            continue
        n = match.group(1)
        voltage = _number(quality.get(f"CH{n}PMTHighVoltage"))
        if voltage is None:
            continue  # a channel the head has but did not image with
        found[str(dye).strip()] = NikonDetector(
            name=f"CH{n}",
            high_voltage=voltage,
            offset=_number(quality.get(f"CH{n}PMTOffset")),
            laser_power=_number(quality.get(f"CH{n}LaserPower")),
        )
    return found


def read_channels(f: Any, tree: Mapping[str, Any] | None = None) -> list[NikonChannel]:
    """One `NikonChannel` per channel of the file, in nd2's channel order."""
    tree = picture_metadata(f) if tree is None else tree
    planes = _items(_get(tree, "PicturePlanes", "PlaneNew"))
    settings = _items(_get(tree, "PicturePlanes", "SampleSetting"))

    channels: list[NikonChannel] = []
    for index, meta in enumerate(f.metadata.channels or []):
        channel = meta.channel
        plane = planes[index] if index < len(planes) else None
        setting_index = _number(_get(plane, "SampleIndex"))
        setting = None
        if settings:
            k = int(setting_index) if setting_index is not None else index
            setting = settings[k] if 0 <= k < len(settings) else settings[0]

        filter_name, lines, band = (None, (), None)
        camera = detector = None
        try:
            if plane is not None:
                filter_name, lines, band = _filter_path(plane)
            if setting is not None:
                camera = _camera(setting)
                detector = _scan_detectors(setting).get(channel.name)
        except Exception:
            logger.exception("Could not read the NIS settings of channel %s", index)

        color = getattr(channel, "color", None)
        channels.append(
            NikonChannel(
                index=index,
                name=_text(channel.name),
                color=(int(color.r), int(color.g), int(color.b)) if color is not None else None,
                emission_nm=_positive(channel.emissionLambdaNm),
                excitation_nm=_positive(channel.excitationLambdaNm),
                modalities=tuple(meta.microscope.modalityFlags or ()),
                pinhole_um=_positive(meta.microscope.pinholeDiameterUm),
                bits=int(meta.volume.bitsPerComponentSignificant)
                if meta.volume.bitsPerComponentSignificant
                else None,
                filter_name=filter_name,
                excitation_lines=lines,
                emission_band=band,
                camera=camera,
                detector=detector,
            )
        )
    return channels


def _stand_name(value: Any) -> str | None:
    """The stand's name; NIS writes a bare number ("1") on some systems, which names nothing."""
    name = _text(value)
    return None if name is None or name.isdigit() else name


def read_scope(f: Any, tree: Mapping[str, Any] | None = None) -> NikonScope:
    """The instrument, from the first channel's typed record and the first sample setting."""
    tree = picture_metadata(f) if tree is None else tree
    channels = f.metadata.channels or []
    first = channels[0] if channels else None
    microscope = first.microscope if first else None
    volume = first.volume if first else None
    setting = next(iter(_items(_get(tree, "PicturePlanes", "SampleSetting"))), None)

    modalities: list[str] = []
    for meta in channels:
        for flag in meta.microscope.modalityFlags or ():
            if flag not in modalities:
                modalities.append(flag)

    z_device = None
    for loop in f.experiment:
        if loop.type == "ZStackLoop":
            z_device = _text(loop.parameters.deviceName)

    def tuple_of(values: Any) -> tuple[float, ...] | None:
        return tuple(float(v) for v in values) if values else None

    return NikonScope(
        microscope=_stand_name(_get(setting, "DeviceSetting", "MicroscopePhysFullName"))
        or _stand_name(_get(setting, "DeviceSetting", "MicroscopeFullName")),
        objective_name=_text(microscope.objectiveName if microscope else None)
        or _text(_get(setting, "ObjectiveSetting", "ObjectiveName")),
        objective_code=_text(_get(setting, "ObjectiveSetting", "ObjectiveCode")),
        magnification=_positive(microscope.objectiveMagnification if microscope else None),
        numerical_aperture=_positive(
            microscope.objectiveNumericalAperture if microscope else None
        ),
        refractive_index=_positive(microscope.immersionRefractiveIndex if microscope else None),
        zoom=_positive(microscope.zoomMagnification if microscope else None),
        projective_magnification=_positive(
            microscope.projectiveMagnification if microscope else None
        ),
        camera_matrix=tuple_of(volume.cameraTransformationMatrix if volume else None),
        pixel_to_stage=tuple_of(volume.pixelToStageTransformationMatrix if volume else None),
        software=_text(f.text_info.get("appVersion")),
        modalities=tuple(modalities),
        camera=_camera(setting) if setting is not None else None,
        z_device=z_device,
    )


def read_calibration(f: Any) -> Calibration:
    """The pixel size, axis by axis, only where NIS calibrated it.

    ``voxel_size()`` answers 1.0 for an axis that is not calibrated, which would be the
    claim that a pixel is a micrometer; ``axesCalibrated`` is what says whether it is.
    """
    channels = f.metadata.channels or []
    if not channels:
        return Calibration(None, None, None)
    volume = channels[0].volume
    calibrated = tuple(volume.axesCalibrated or (False, False, False))
    values = tuple(volume.axesCalibration or (0.0, 0.0, 0.0))

    def axis(i: int) -> float | None:
        return _positive(values[i]) if i < len(calibrated) and calibrated[i] else None

    z = axis(2)
    stack = z_stack(f)
    if stack is not None and stack.step_um is not None:
        z = stack.step_um
    return Calibration(axis(0), axis(1), z)


def z_stack(f: Any) -> ZStack | None:
    for loop in f.experiment:
        if loop.type == "ZStackLoop" and loop.count > 1:
            params = loop.parameters
            return ZStack(
                count=int(loop.count),
                step_um=_positive(abs(params.stepUm)) if params.stepUm else None,
                bottom_to_top=bool(params.bottomToTop),
                home_index=int(params.homeIndex) if params.homeIndex is not None else None,
                device=_text(params.deviceName),
            )
    return None


def stage_xy(x: float | None, y: float | None) -> tuple[float | None, float | None]:
    """A stage position, or ``(None, None)`` when the stage was not read out.

    A system without an encoded stage writes exactly (0, 0) for every point — the Ti2
    simulator's ``t3p3z5c3.nd2`` puts all three of its positions there. Read as
    positions, every field would be placed on top of every other.
    """
    if x is None or y is None or (x == 0 and y == 0):
        return None, None
    return x, y


def read_positions(f: Any, frames: FrameTable | None = None) -> list[Position]:
    """One `Position` per point along ``P``, or the single field when there is no ``P``.

    From each point's first frame, not from the loop's list of points: the frame records
    where the stage actually was (and the name), the list where it was meant to go.
    """
    count = int(f.sizes.get("P", 1))
    if "P" in f.sizes and frames is None:
        frames = FrameTable.of_loops(list(f.loop_indices))
    found: list[Position] = []
    for index in range(count):
        seq: int | None = 0
        if "P" in f.sizes:
            seqs = frames.select(P=index)
            seq = int(seqs[0]) if len(seqs) else None
        x = y = focus = None
        name = None
        if seq is not None:
            try:
                frame = f.frame_metadata(seq).channels[0]
                stage = frame.position.stagePositionUm
                x, y = stage_xy(_number(stage.x), _number(stage.y))
                focus = _number(stage.z)
                name = _text(frame.position.name)
            except Exception:
                logger.exception("Could not read where position %s was", index)
        found.append(Position(index if "P" in f.sizes else None, name, x, y, focus))
    return found


# ---------------------------------------------------------------------------
# The frame table: when each frame was taken, and at what focus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTable:
    """Per sequence index: its loop coordinates, its time (UTC, or NaT) and its z (µm,
    or NaN where NIS did not record a per-plane z).

    The coordinates are one integer array per loop axis (``T``, ``P``, ``Z`` …), so that
    picking a point's frames out of a 96-well × 500-timepoint run is one vectorised
    comparison, not a pass over 100k dicts per call.
    """

    axes: Mapping[str, np.ndarray]
    times: np.ndarray
    z_um: np.ndarray

    @classmethod
    def of_loops(
        cls,
        loops: Sequence[Mapping[str, int]],
        times: np.ndarray | None = None,
        z_um: np.ndarray | None = None,
    ) -> FrameTable:
        count = len(loops)
        names = sorted({axis for indices in loops for axis in indices})
        axes = {
            axis: np.fromiter((indices.get(axis, 0) for indices in loops), dtype=np.int64, count=count)
            for axis in names
        }
        return cls(
            axes,
            np.full(count, np.datetime64("NaT", "us")) if times is None else np.asarray(times, dtype="datetime64[us]"),
            np.full(count, np.nan) if z_um is None else np.asarray(z_um, dtype=float),
        )

    def __len__(self) -> int:
        return len(self.times)

    def axis(self, name: str) -> np.ndarray:
        """Each frame's index along `name`; 0 for an axis the run did not loop over."""
        found = self.axes.get(name)
        return found if found is not None else np.zeros(len(self), dtype=np.int64)

    def select(self, **coords: int) -> np.ndarray:
        """The sequence indices at these loop coordinates, in order."""
        mask = np.ones(len(self), dtype=bool)
        for axis, value in coords.items():
            mask &= self.axis(axis) == value
        return np.flatnonzero(mask)


def jdn_to_datetime64(jdn: float) -> np.datetime64:
    seconds = (jdn - UNIX_EPOCH_JDN) * 86400.0
    return np.datetime64(round(seconds * 1e6), "us")


def read_frame_table(f: Any) -> FrameTable:
    """The frame table, from ``events()`` and the first frame's absolute stamp.

    ``events()`` is the "Recorded Data" table: per frame, seconds since the run began and
    — on a system that reads its focus out — the absolute ``Z Coord [µm]``. Rows are keyed
    by their ``Index`` (the sequence index); rows for events that are not frames are
    skipped. The absolute time of frame 0 anchors the relative ones. Should the table
    not cover every frame, the times come from each frame's metadata instead.
    """
    loops = list(f.loop_indices)
    count = len(loops)
    times = np.full(count, np.datetime64("NaT", "us"))
    z_um = np.full(count, np.nan)

    relative = np.full(count, np.nan)
    try:
        table = f.events(orient="list")
    except Exception:
        logger.exception("Could not read the ND2 events table")
        table = {}
    index = list(table.get("Index", []))
    seconds = list(table.get("Time [s]", []))
    z_column = list(table.get("Z Coord [µm]", []))
    for row, seq in enumerate(index):
        seq_number = _number(seq)
        if seq_number is None or not 0 <= int(seq_number) < count:
            continue
        seq = int(seq_number)
        if row < len(seconds) and _number(seconds[row]) is not None:
            relative[seq] = float(seconds[row])
        if row < len(z_column) and _number(z_column[row]) is not None:
            z_um[seq] = float(z_column[row])

    try:
        first = f.frame_metadata(0).channels[0].time
        if count and _positive(first.absoluteJulianDayNumber) and not np.isnan(relative).any():
            # Frame 0 is where its own stamp says; every other frame its recorded offset
            # from frame 0 later. In whole microseconds from one anchor: a fraction of a
            # day added to a JDN of 2.46 million keeps only ~5 µs.
            anchor = jdn_to_datetime64(first.absoluteJulianDayNumber)
            offsets_us = np.round((relative - relative[0]) * 1e6).astype("int64")
            times = anchor + offsets_us.astype("timedelta64[us]")
        else:
            for seq in range(count):
                stamp = f.frame_metadata(seq).channels[0].time.absoluteJulianDayNumber
                if _positive(stamp):
                    times[seq] = jdn_to_datetime64(stamp)
    except Exception:
        logger.exception("Could not read the ND2 frame times")

    return FrameTable.of_loops(loops, times, z_um)


# ---------------------------------------------------------------------------
# The record on the image anchor
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Anything nd2 hands back, as JSON: dataclasses, named tuples, enums, numpy."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return {k: _plain(v) for k, v in value._asdict().items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def image_record(
    f: Any,
    file_name: str,
    file_id: str,
    position: Position,
    rgb_channel: int | None = None,
) -> str:
    """The JSON the image anchor's metadata blob takes: NIS's own record of the run.

    ``OmeMetadataInput`` is the only document slot an anchor has, and it takes any JSON
    string. What goes in here is not OME, and says so under ``format``: nd2's typed
    metadata, the experiment's loops, the attributes and NIS's text info, verbatim.
    """
    record = {
        "format": "nikon-nd2",
        "source_file": file_name,
        "source_file_id": file_id,
        "position_index": position.index,
        "position_name": position.name,
        "rgb_channel": rgb_channel,
        "attributes": f.attributes,
        "text_info": f.text_info,
        "experiment": f.experiment,
        "metadata": f.metadata,
    }
    return json.dumps(_plain(record))


__all__ = [
    "Calibration",
    "FrameTable",
    "NikonCamera",
    "NikonChannel",
    "NikonDetector",
    "NikonScope",
    "Position",
    "ZStack",
    "image_record",
    "jdn_to_datetime64",
    "nm",
    "picture_metadata",
    "read_calibration",
    "read_channels",
    "read_frame_table",
    "read_positions",
    "read_scope",
    "stage_xy",
    "um",
    "z_stack",
]

