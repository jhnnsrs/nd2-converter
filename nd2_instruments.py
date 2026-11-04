"""Nikon instrument models: what kind of microscope recorded a file, and its light path.

NIS-Elements tags every channel with modality flags (``camera``, ``fluorescence``,
``laserScanConfocal``, ``spinningDiskConfocal``, ``gaasp``, ``TIRF``, …), and those decide
the model. Unlike a LIF there is no system name to match: the flags are the only
statement of how the instrument is built.

| Model | Flags | Light path |
|---|---|---|
| ``nikon.point-scanning`` | laserScanConfocal, pmt, gaasp, multiphoton, nonDescannedDetector | lines → scanner → objective → sample → objective → scanner → pinhole → band → PMT/GaAsP. A non-descanned detector takes the light off the objective, a transmitted one sits behind the sample. |
| ``nikon.spinning-disk`` | spinningDiskConfocal, sora, dsdConfocal, sweptField…, RCM, iSIM, liveSR | lines → confocal unit → objective → sample → objective → confocal unit → camera |
| ``nikon.tirf`` | TIRF | lines → objective → sample → objective → camera |
| ``nikon.sim`` | SIM | lines → structured illumination → objective → sample → objective → camera |
| ``nikon.widefield`` | camera, fluorescence, brightfield, … | light source → [filter] → objective → sample → objective → [filter] → camera |
| ``nikon.generic`` | none | what the file names, nothing more |

The first matching row wins, in that order. A channel whose own flags say transmitted
light (brightfield, DIC, phase) is drawn as a transmitted path on any model: a lamp
behind the sample, then the objective, then the camera (or the transmitted detector of a
point scanner).

Recorded and modelled are kept apart, as in the LIF converter. An element the file names
(a laser line, the camera, the detector, the pinhole, a named filter) is recorded. One
that only the instrument's design implies (the scanner, the confocal unit, the light
source of a widefield channel) is labelled "(implied)" and never carries a number.

mikro has no instrument entity, so the identification rides on the ``microscope``
device as settings: ``instrument-model``, ``family``, ``modalities``, the stand as
``model``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mikro.api.schema import (
    BeamSplitterElementInput,
    BeamStateInput,
    CCDElementInput,
    ChannelKind,
    DetectorElementInput,
    FilterElementInput,
    FilterKind,
    LampElementInput,
    LaserElementInput,
    LightEdgeInput,
    LightpathGraphInput,
    LightPortInput,
    MirrorElementInput,
    ObjectiveElementInput,
    OpticalElementInput,
    OptikitStateInput,
    PinholeElementInput,
    PortRole,
    SampleElementInput,
    SpectrumInput,
)
from mikro.inputs.lightpath import in_out as _in_out
from mikro.inputs.lightpath import port as _port

from converter_metadata import device, devices, setting
from nd2_metadata import NikonChannel, NikonScope, nm, um

Hop = tuple[str, str, str, str, BeamStateInput | None, str]


# ---------------------------------------------------------------------------
# Building a light path
# ---------------------------------------------------------------------------


@dataclass
class _Path:
    """Elements and the hops between them, built together so that no edge can name a
    port its element does not declare (nothing checks that before the server does)."""

    model: str
    elements: list[OpticalElementInput] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)

    def add(self, element: OpticalElementInput) -> OpticalElementInput:
        self.elements.append(element)
        return element

    def hop(
        self,
        source: tuple[str, str] | None,
        target: tuple[str, str],
        beam: BeamStateInput | None,
        medium: str = "AIR",
    ) -> None:
        if source is not None:
            self.hops.append((*source, *target, beam, medium))

    def implied(self, label: str) -> str:
        return f"{label} (implied)"

    @property
    def implied_note(self) -> str:
        return f"implied by {self.model}"

    def graph(self) -> LightpathGraphInput | None:
        if len(self.elements) < 2 or not self.hops:
            return None
        return LightpathGraphInput(
            elements=self.elements,
            edges=[
                LightEdgeInput(
                    id=f"{src}:{src_port}->{dst}:{dst_port}",
                    source_element_id=src,
                    source_port_id=src_port,
                    target_element_id=dst,
                    target_port_id=dst_port,
                    beam=beam,
                    medium=medium,
                )
                for src, src_port, dst, dst_port, beam, medium in self.hops
            ],
        )


def _four_ports(band: SpectrumInput | None = None) -> list[LightPortInput]:
    """An element both beams cross: excitation one way, emission the other."""
    return [
        _port("ex-in", "excitation in", PortRole.INPUT),
        _port("ex-out", "excitation out", PortRole.OUTPUT),
        _port("det-in", "emission in", PortRole.INPUT),
        _port("det-out", "emission out", PortRole.OUTPUT, band),
    ]


def _band(channel: NikonChannel) -> SpectrumInput | None:
    if channel.emission_band is None:
        return None
    low, high = channel.emission_band
    return SpectrumInput(min=nm(low), max=nm(high))


def _emission_nm(channel: NikonChannel) -> float | None:
    if channel.emission_band is not None:
        return sum(channel.emission_band) / 2
    return channel.emission_nm


def _excitation_nm(channel: NikonChannel) -> float | None:
    if len(channel.excitation_lines) == 1:
        return channel.excitation_lines[0]
    if not channel.excitation_lines:
        return channel.excitation_nm
    return None


def _objective(scope: NikonScope) -> ObjectiveElementInput | None:
    if not (scope.objective_name or scope.numerical_aperture or scope.magnification):
        return None
    return ObjectiveElementInput(
        id="objective",
        label=scope.objective_name or "Objective",
        brand="Nikon",
        model=scope.objective_name,
        serial_number=scope.objective_code,
        magnification=scope.magnification,
        numerical_aperture=scope.numerical_aperture,
        immersion_medium=scope.immersion,
        ports=[
            _port("illum-in", "excitation in", PortRole.INPUT),
            _port("illum-out", "excitation to sample", PortRole.OUTPUT),
            _port("det-in", "light from the sample", PortRole.INPUT),
            _port("det-out", "light out", PortRole.OUTPUT),
        ],
    )


def _medium(scope: NikonScope) -> str:
    immersion = scope.immersion
    return immersion.value if immersion is not None else "AIR"


def _lines(path: _Path, channel: NikonChannel) -> tuple[str, str] | None:
    """Every laser line the filter path lists, combined into one beam; the tail of it."""
    lines = channel.excitation_lines or (
        (channel.excitation_nm,) if channel.excitation_nm is not None else ()
    )
    for i, wavelength in enumerate(lines):
        path.add(
            LaserElementInput(
                id=f"line-{i}",
                label=f"Laser {wavelength:g} nm",
                nominal_wavelength=nm(wavelength),
                channel=ChannelKind.FREE_SPACE,
                ports=[_port("out", "output", PortRole.OUTPUT)],
            )
        )
    if not lines:
        return None
    if len(lines) == 1:
        return ("line-0", "out")
    ports = [_port(f"in-{i}", f"line {i}", PortRole.INPUT) for i in range(len(lines))]
    ports.append(_port("out", "combined", PortRole.OUTPUT))
    path.add(
        BeamSplitterElementInput(
            id="combiner", label=path.implied("Laser combiner"), ports=ports
        )
    )
    for i in range(len(lines)):
        path.hop((f"line-{i}", "out"), ("combiner", f"in-{i}"), None)
    return ("combiner", "out")


def _camera(path: _Path, scope: NikonScope, channel: NikonChannel, resolution) -> str:
    camera = channel.camera or scope.camera
    path.add(
        CCDElementInput(
            id="camera",
            label=(camera.name if camera else None) or path.implied("Camera"),
            model=camera.family if camera else None,
            resolution=list(resolution) if resolution else None,
            ports=[_port("in", "sensor", PortRole.INPUT, _band(channel))],
        )
    )
    return "camera"


def _sample(path: _Path, channel: NikonChannel, sample_label: str) -> None:
    path.add(
        SampleElementInput(
            id="sample",
            label=sample_label,
            description=f"imaged as {channel.name}" if channel.name else None,
            ports=[
                _port("illum-in", "illuminated field", PortRole.INPUT),
                _port(
                    "out",
                    "transmitted light" if channel.transmitted else "emitted light",
                    PortRole.OUTPUT,
                    None if channel.transmitted else _band(channel),
                ),
            ],
        )
    )


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NikonInstrument:
    """An instrument model. This base is the generic one: only what the file names."""

    key: str = "nikon.generic"
    family: str = "Nikon"
    flags: tuple[str, ...] = ()
    #: Whether its fluorescence channels are excited by laser lines rather than a lamp.
    lasers: bool = False

    def matches(self, scope: NikonScope) -> bool:
        return bool(set(self.flags) & set(scope.modalities))

    # -- devices --------------------------------------------------------------

    def identity(self, scope: NikonScope):
        return device(
            "microscope",
            "microscope",
            setting("manufacturer", string="Nikon"),
            setting("model", string=scope.microscope),
            setting("instrument-model", string=self.key),
            setting("family", string=self.family),
            setting("modalities", string=", ".join(scope.modalities) or None),
            setting("software", string=scope.software),
            setting("z-device", string=scope.z_device),
        )

    def devices(self, scope: NikonScope) -> list:
        return devices(
            self.identity(scope),
            device(
                "objective",
                "objective",
                setting("model", string=scope.objective_name),
                setting("order-number", string=scope.objective_code),
                setting("magnification", number=scope.magnification),
                setting("numerical-aperture", number=scope.numerical_aperture),
                setting("immersion", string=scope.immersion.value if scope.immersion else None),
                setting("refractive-index", number=scope.refractive_index),
                setting("zoom", number=scope.zoom),
                setting("projective-magnification", number=scope.projective_magnification),
            ),
            self.camera_device(scope),
            # How the camera's axes sit against the stage's. Recorded, not applied: a
            # placement by stage centre has no rotation to put it in.
            device(
                "camera-orientation",
                "transform",
                setting("rotation-deg", number=scope.camera_rotation_deg),
                setting(
                    "camera-matrix",
                    string=" ".join(f"{v:g}" for v in scope.camera_matrix)
                    if scope.camera_matrix
                    else None,
                ),
                setting(
                    "pixel-to-stage",
                    string=" ".join(f"{v:g}" for v in scope.pixel_to_stage)
                    if scope.pixel_to_stage
                    else None,
                ),
            ),
        )

    def camera_device(self, scope: NikonScope):
        camera = scope.camera
        if camera is None:
            return None
        return device(
            "camera",
            "camera",
            setting("name", string=camera.name),
            setting("model", string=camera.family),
            setting("binning", string=camera.binning),
            setting("format", string=camera.format),
        )

    def channel_devices(self, channel: NikonChannel) -> OptikitStateInput | None:
        index = channel.index
        detector = channel.detector
        found = devices(
            device(
                f"channel-{index}",
                "channel",
                setting("name", string=channel.name),
                setting(
                    "color",
                    string="#{:02x}{:02x}{:02x}".format(*channel.color)
                    if channel.color
                    else None,
                ),
                setting("modalities", string=", ".join(channel.modalities) or None),
                setting("bits", number=float(channel.bits) if channel.bits else None),
                setting("emission", quantity=nm(channel.emission_nm)),
                setting("excitation", quantity=nm(channel.excitation_nm)),
            ),
            device(
                f"filter-{index}",
                "filter",
                setting("name", string=channel.filter_name),
                setting(
                    "excitation-lines",
                    string=", ".join(f"{line:g} nm" for line in channel.excitation_lines)
                    or None,
                ),
                setting(
                    "emission-band",
                    string="{:g}-{:g} nm".format(*channel.emission_band)
                    if channel.emission_band
                    else None,
                ),
            ),
            self.channel_acquisition(channel),
            device(
                f"detector-{index}",
                "detector",
                setting("name", string=detector.name if detector else None),
                setting("high-voltage", number=detector.high_voltage if detector else None),
                setting("offset", number=detector.offset if detector else None),
                setting("laser-power", number=detector.laser_power if detector else None),
            ),
            device(
                f"pinhole-{index}",
                "pinhole",
                setting("diameter", quantity=um(channel.pinhole_um)),
            )
            if self.has_pinhole
            else None,
        )
        return OptikitStateInput(devices=found) if found else None

    def channel_acquisition(self, channel: NikonChannel):
        camera = channel.camera
        if camera is None:
            return None
        return device(
            f"camera-{channel.index}",
            "camera",
            setting("name", string=camera.name),
            setting(
                "exposure",
                quantity=f"{camera.exposure_ms:g} ms" if camera.exposure_ms else None,
            ),
            setting("binning", string=camera.binning),
        )

    @property
    def has_pinhole(self) -> bool:
        return False

    # -- the light path ---------------------------------------------------------

    def channel_lightpath(
        self,
        channel: NikonChannel,
        scope: NikonScope,
        sample_label: str,
        resolution: Sequence[int] | None = None,
    ) -> LightpathGraphInput | None:
        path = _Path(self.key)
        if channel.transmitted:
            return self.transmitted_path(path, channel, scope, sample_label, resolution)
        return self.fluorescence_path(path, channel, scope, sample_label, resolution)

    def transmitted_path(
        self,
        path: _Path,
        channel: NikonChannel,
        scope: NikonScope,
        sample_label: str,
        resolution: Sequence[int] | None,
    ) -> LightpathGraphInput | None:
        """Lamp → sample → objective → camera: light through the sample, not from it."""
        path.add(
            LampElementInput(
                id="source",
                label=path.implied("Transmitted light"),
                channel=ChannelKind.FREE_SPACE,
                ports=[_port("out", "output", PortRole.OUTPUT)],
            )
        )
        _sample(path, channel, sample_label)
        path.hop(("source", "out"), ("sample", "illum-in"), None)
        tail = ("sample", "out")
        objective = _objective(scope)
        if objective is not None:
            path.add(objective)
            path.hop(tail, ("objective", "det-in"), None, _medium(scope))
            tail = ("objective", "det-out")
        target = self.transmitted_target(path, channel, scope, resolution)
        path.hop(tail, (target, "in"), None)
        return path.graph()

    def transmitted_target(self, path, channel, scope, resolution) -> str:
        return _camera(path, scope, channel, resolution)

    def fluorescence_path(
        self,
        path: _Path,
        channel: NikonChannel,
        scope: NikonScope,
        sample_label: str,
        resolution: Sequence[int] | None,
    ) -> LightpathGraphInput | None:
        """Source → [filter] → objective → sample → objective → [filter] → camera."""
        excitation = BeamStateInput(wavelength=nm(_excitation_nm(channel)))
        emission = BeamStateInput(wavelength=nm(_emission_nm(channel)))
        medium = _medium(scope)

        tail = _lines(path, channel) if self.lasers else None
        if tail is None:
            path.add(
                LampElementInput(
                    id="source",
                    label=path.implied("Light source"),
                    channel=ChannelKind.FREE_SPACE,
                    ports=[_port("out", "output", PortRole.OUTPUT)],
                )
            )
            tail = ("source", "out")

        tail, after = self.excitation_optics(path, channel, tail, excitation)

        objective = _objective(scope)
        if objective is not None:
            path.add(objective)
            path.hop(tail, ("objective", "illum-in"), excitation)
            tail = ("objective", "illum-out")
        _sample(path, channel, sample_label)
        path.hop(tail, ("sample", "illum-in"), excitation, medium)
        tail = ("sample", "out")
        if objective is not None:
            path.hop(tail, ("objective", "det-in"), emission, medium)
            tail = ("objective", "det-out")
        for element in after:
            path.hop(tail, (element, "det-in"), emission)
            tail = (element, "det-out")
        tail = self.emission_optics(path, channel, tail, emission)
        target = self.detector(path, channel, scope, resolution)
        path.hop(tail, (target, "in"), emission)
        return path.graph()

    def excitation_optics(
        self, path: _Path, channel: NikonChannel, tail, beam
    ) -> tuple[tuple[str, str], list[str]]:
        """What sits between the source and the objective, crossed again on the way back.

        Returns the new tail and the elements the emission crosses in reverse order.
        """
        if channel.filter_name:
            path.add(
                FilterElementInput(
                    id="filter",
                    label=channel.filter_name,
                    model=channel.filter_name,
                    filter_kind=FilterKind.DICHROIC,
                    ports=_four_ports(_band(channel)),
                )
            )
            path.hop(tail, ("filter", "ex-in"), beam)
            return ("filter", "ex-out"), ["filter"]
        return tail, []

    def emission_optics(self, path: _Path, channel: NikonChannel, tail, beam):
        return tail

    def detector(self, path, channel, scope, resolution) -> str:
        return _camera(path, scope, channel, resolution)


@dataclass(frozen=True)
class Widefield(NikonInstrument):
    """A camera on a widefield stand: Ti/Ti2/Eclipse with a lamp or LED."""

    key: str = "nikon.widefield"
    family: str = "Widefield"
    flags: tuple[str, ...] = (
        "camera",
        "fluorescence",
        "brightfield",
        "diContrast",
        "phaseContrast",
    )


@dataclass(frozen=True)
class Tirf(NikonInstrument):
    """TIRF: laser lines brought to the objective's edge, a camera behind it."""

    key: str = "nikon.tirf"
    family: str = "TIRF"
    flags: tuple[str, ...] = ("TIRF",)
    lasers: bool = True


@dataclass(frozen=True)
class Sim(NikonInstrument):
    """N-SIM: laser lines through a structured-illumination unit, a camera behind it."""

    key: str = "nikon.sim"
    family: str = "SIM"
    flags: tuple[str, ...] = ("SIM",)
    lasers: bool = True

    def excitation_optics(self, path, channel, tail, beam):
        path.add(
            FilterElementInput(
                id="sim",
                label=path.implied("Structured illumination"),
                description=path.implied_note,
                filter_kind=FilterKind.OTHER,
                ports=_in_out(),
            )
        )
        path.hop(tail, ("sim", "in"), beam)
        tail, after = super().excitation_optics(path, channel, ("sim", "out"), beam)
        return tail, after


@dataclass(frozen=True)
class SpinningDisk(NikonInstrument):
    """A camera-based confocal: a spinning disk (CSU, X-Light, SoRa), a swept field, or
    a re-scan unit (RCM, iSIM, Live-SR). Excitation and emission both cross the unit."""

    key: str = "nikon.spinning-disk"
    family: str = "Spinning-disk confocal"
    flags: tuple[str, ...] = (
        "spinningDiskConfocal",
        "sora",
        "dsdConfocal",
        "sweptFieldConfocalPinhole",
        "sweptFieldConfocalSlit",
        "RCM",
        "iSIM",
        "liveSR",
    )
    lasers: bool = True

    @property
    def has_pinhole(self) -> bool:
        return True

    def unit_label(self, channel: NikonChannel) -> str:
        flags = set(channel.modalities)
        if "sora" in flags:
            return "SoRa disk"
        if flags & {"sweptFieldConfocalPinhole", "sweptFieldConfocalSlit"}:
            return "Swept-field unit"
        if "RCM" in flags:
            return "Re-scan unit"
        if flags & {"iSIM", "liveSR"}:
            return "Instant SIM unit"
        return "Spinning disk"

    def excitation_optics(self, path, channel, tail, beam):
        label = self.unit_label(channel)
        path.add(
            PinholeElementInput(
                id="disk",
                label=f"{label} ({channel.pinhole_um:g} um pinholes)"
                if channel.pinhole_um
                else path.implied(label),
                diameter=um(channel.pinhole_um),
                ports=_four_ports(),
            )
        )
        path.hop(tail, ("disk", "ex-in"), beam)
        tail, after = super().excitation_optics(path, channel, ("disk", "ex-out"), beam)
        return tail, [*after, "disk"]


@dataclass(frozen=True)
class PointScanning(NikonInstrument):
    """A point scanner (A1, AX, C2): lines → scanner → objective → sample, and back
    through the scanner → pinhole → detection band → PMT or GaAsP detector. A
    non-descanned detector takes the light off the objective; a transmitted one
    (``transmitDetector``) sits behind the sample."""

    key: str = "nikon.point-scanning"
    family: str = "Point-scanning confocal"
    flags: tuple[str, ...] = (
        "laserScanConfocal",
        "pmt",
        "gaasp",
        "multiphoton",
        "nonDescannedDetector",
    )
    lasers: bool = True

    @property
    def has_pinhole(self) -> bool:
        return True

    def camera_device(self, scope: NikonScope):
        # On a point scanner NIS's "camera" is the scan head.
        camera = scope.camera
        if camera is None:
            return None
        return device(
            "scan-head",
            "scanner",
            setting("name", string=camera.name),
            setting("model", string=camera.family),
            setting("zoom", number=scope.zoom),
        )

    def channel_acquisition(self, channel: NikonChannel):
        camera = channel.camera
        if camera is None:
            return None
        return device(
            f"scan-{channel.index}",
            "scanner",
            setting(
                "frame-time",
                quantity=f"{camera.exposure_ms:g} ms" if camera.exposure_ms else None,
            ),
        )

    def descanned(self, channel: NikonChannel) -> bool:
        return not ({"multiphoton", "nonDescannedDetector"} & set(channel.modalities))

    def excitation_optics(self, path, channel, tail, beam):
        path.add(
            MirrorElementInput(
                id="scanner", label=path.implied("Scanner"), ports=_four_ports()
            )
        )
        path.hop(tail, ("scanner", "ex-in"), beam)
        return ("scanner", "ex-out"), (["scanner"] if self.descanned(channel) else [])

    def emission_optics(self, path, channel, tail, beam):
        if self.descanned(channel) and channel.pinhole_um is not None:
            path.add(
                PinholeElementInput(
                    id="pinhole",
                    label="Confocal pinhole",
                    diameter=um(channel.pinhole_um),
                    ports=_in_out(),
                )
            )
            path.hop(tail, ("pinhole", "in"), beam)
            tail = ("pinhole", "out")
        band = _band(channel)
        if band is not None or channel.filter_name:
            path.add(
                FilterElementInput(
                    id="emission-filter",
                    label=channel.filter_name or "Emission filter",
                    filter_kind=FilterKind.BAND_PASS,
                    ports=_in_out(band),
                )
            )
            path.hop(tail, ("emission-filter", "in"), beam)
            tail = ("emission-filter", "out")
        return tail

    def detector_model(self, channel: NikonChannel) -> str | None:
        flags = set(channel.modalities)
        if "gaasp" in flags:
            return "GaAsP"
        if "pmt" in flags:
            return "PMT"
        return None

    def detector(self, path, channel, scope, resolution) -> str:
        detector = channel.detector
        path.add(
            DetectorElementInput(
                id="detector",
                label=(detector.name if detector else None)
                or path.implied(self.detector_model(channel) or "Detector"),
                manufacturer="Nikon",
                model=self.detector_model(channel),
                ports=[_port("in", "sensor", PortRole.INPUT, _band(channel))],
            )
        )
        return "detector"

    def transmitted_target(self, path, channel, scope, resolution) -> str:
        path.add(
            DetectorElementInput(
                id="detector",
                label=path.implied("Transmitted light detector"),
                manufacturer="Nikon",
                ports=[_port("in", "sensor", PortRole.INPUT)],
            )
        )
        return "detector"

    def transmitted_path(self, path, channel, scope, sample_label, resolution):
        """A point scanner's transmitted channel: the scanned beam through the sample
        onto the transmitted detector — the lines, not a lamp, light it."""
        tail = _lines(path, channel)
        if tail is None:
            return super().transmitted_path(path, channel, scope, sample_label, resolution)
        tail, _ = self.excitation_optics(path, channel, tail, None)
        objective = _objective(scope)
        if objective is not None:
            path.add(objective)
            path.hop(tail, ("objective", "illum-in"), None)
            tail = ("objective", "illum-out")
        _sample(path, channel, sample_label)
        path.hop(tail, ("sample", "illum-in"), None, _medium(scope))
        target = self.transmitted_target(path, channel, scope, resolution)
        path.hop(("sample", "out"), (target, "in"), None)
        return path.graph()


GENERIC = NikonInstrument()
# The most specific first: a spinning-disk system's channels are also "camera" and
# "fluorescence", a point scanner's TD channel also "brightfield".
INSTRUMENTS: tuple[NikonInstrument, ...] = (
    PointScanning(),
    SpinningDisk(),
    Sim(),
    Tirf(),
    Widefield(),
)


def identify(scope: NikonScope) -> NikonInstrument:
    """The model for this file's modality flags; the generic one when none matches."""
    for instrument in INSTRUMENTS:
        if instrument.matches(scope):
            return instrument
    return GENERIC


@dataclass(frozen=True)
class NikonAcquisition:
    """Everything one file says about the instrument, and the model that reads it."""

    instrument: NikonInstrument
    scope: NikonScope

    def devices(self) -> list:
        return self.instrument.devices(self.scope)

    def channel_devices(self, channel: NikonChannel) -> OptikitStateInput | None:
        return self.instrument.channel_devices(channel)

    def channel_lightpath(
        self,
        channel: NikonChannel,
        sample_label: str,
        resolution: Sequence[int] | None = None,
    ) -> LightpathGraphInput | None:
        return self.instrument.channel_lightpath(
            channel, self.scope, sample_label, resolution
        )


def acquisition_of(scope: NikonScope) -> NikonAcquisition:
    return NikonAcquisition(identify(scope), scope)
