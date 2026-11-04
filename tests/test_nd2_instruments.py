"""Offline checks of the instrument models: which model a file's flags pick, and the
light path each lays out.

The channels are shaped like the sample files' (``cluster.nd2`` for the A1,
``001-overview_crop.nd2`` for the spinning disk, ``t3p3z5c3.nd2`` for the Ti2 widefield).
"""

import pytest

import nd2_instruments as I
from nd2_metadata import NikonCamera, NikonChannel, NikonDetector, NikonScope

OBJECTIVE = dict(objective_name="Plan Apo λ 60x Oil", magnification=60.0, numerical_aperture=1.4)


def a1_channel(**kw):
    values = dict(
        index=0,
        name="5-FAM/pH 9.0",
        color=(120, 233, 15),
        emission_nm=525.0,
        excitation_nm=488.0,
        modalities=("fluorescence", "laserScanConfocal", "gaasp"),
        pinhole_um=38.3,
        excitation_lines=(488.0, 561.0),
        emission_band=(500.0, 550.0),
        camera=NikonCamera(name="Nikon A1plus", family="Nikon A1", exposure_ms=997.0),
        detector=NikonDetector("CH2", 37.0, -29.0, 0.5),
    )
    values.update(kw)
    return NikonChannel(**values)


def chain(graph):
    """The element ids in the order the light first reaches them, following the hops in
    the order they were laid: an element crossed twice (the objective) counts once."""
    order: list[str] = []
    for edge in graph.edges:
        if edge.source_element_id not in order:
            order.append(edge.source_element_id)
    last = graph.edges[-1].target_element_id
    return order + ([last] if last not in order else [])


def elements(graph):
    return {e.id: e for e in graph.elements}


def check_ports(graph):
    """Every edge names a port its element declares (the server rejects anything else)."""
    ports = {e.id: {p.id for p in e.ports} for e in graph.elements}
    for edge in graph.edges:
        assert edge.source_port_id in ports[edge.source_element_id], edge.id
        assert edge.target_port_id in ports[edge.target_element_id], edge.id


@pytest.mark.parametrize(
    ("flags", "key"),
    [
        (("fluorescence", "laserScanConfocal", "pmt"), "nikon.point-scanning"),
        (("fluorescence", "multiphoton"), "nikon.point-scanning"),
        (("brightfield", "spinningDiskConfocal"), "nikon.spinning-disk"),
        (("fluorescence", "camera", "sora"), "nikon.spinning-disk"),
        (("fluorescence", "camera", "SIM"), "nikon.sim"),
        (("fluorescence", "camera", "TIRF"), "nikon.tirf"),
        (("fluorescence", "camera"), "nikon.widefield"),
        (("brightfield",), "nikon.widefield"),
        ((), "nikon.generic"),
    ],
)
def test_the_flags_pick_the_model(flags, key):
    assert I.identify(NikonScope(modalities=flags)).key == key


def test_an_a1_channel_is_scanned_pinholed_and_detected():
    scope = NikonScope(modalities=("fluorescence", "laserScanConfocal", "gaasp"), **OBJECTIVE)
    graph = I.acquisition_of(scope).channel_lightpath(a1_channel(), "Sample")
    check_ports(graph)
    assert chain(graph) == [
        "line-0", "line-1", "combiner", "scanner", "objective", "sample", "pinhole", "emission-filter", "detector",
    ]
    found = elements(graph)
    assert found["detector"].label == "CH2" and found["detector"].model == "GaAsP"
    assert found["scanner"].label.endswith("(implied)")
    assert found["combiner"].label.endswith("(implied)")
    assert not found["detector"].label.endswith("(implied)")


def test_a_non_descanned_detector_takes_the_light_off_the_objective():
    scope = NikonScope(modalities=("fluorescence", "multiphoton"), **OBJECTIVE)
    channel = a1_channel(modalities=("fluorescence", "multiphoton"), excitation_lines=(920.0,))
    graph = I.acquisition_of(scope).channel_lightpath(channel, "Sample")
    check_ports(graph)
    assert "pinhole" not in elements(graph)
    # Out of the objective, past the scanner, straight to the band and the detector.
    assert ("objective", "emission-filter") in {(e.source_element_id, e.target_element_id) for e in graph.edges}


def test_a_transmitted_a1_channel_ends_on_the_transmitted_detector():
    scope = NikonScope(modalities=("laserScanConfocal", "transmitDetector"), **OBJECTIVE)
    channel = a1_channel(modalities=("transmitDetector",), name="TD", emission_band=None)
    graph = I.acquisition_of(scope).channel_lightpath(channel, "Sample")
    check_ports(graph)
    assert chain(graph)[-2:] == ["sample", "detector"]
    assert elements(graph)["detector"].label == "Transmitted light detector (implied)"


def test_a_spinning_disk_channel_crosses_the_disk_both_ways():
    scope = NikonScope(modalities=("fluorescence", "spinningDiskConfocal"), **OBJECTIVE)
    channel = NikonChannel(
        0, "GFP", (85, 255, 0), 525.0, None,
        modalities=("fluorescence", "spinningDiskConfocal"),
        pinhole_um=50.0,
        excitation_lines=(488.0,),
        camera=NikonCamera(name="Andor DU-888 X-12403"),
    )
    graph = I.acquisition_of(scope).channel_lightpath(channel, "Sample", (1024, 1024))
    check_ports(graph)
    hops = [(e.source_element_id, e.target_element_id) for e in graph.edges]
    assert ("line-0", "disk") in hops and ("objective", "disk") in hops and ("disk", "camera") in hops
    found = elements(graph)
    assert found["disk"].label == "Spinning disk (50 um pinholes)"
    assert found["camera"].label == "Andor DU-888 X-12403"
    assert list(found["camera"].resolution) == [1024, 1024]


def test_a_brightfield_channel_on_a_spinning_disk_is_transmitted():
    scope = NikonScope(modalities=("brightfield", "spinningDiskConfocal"), **OBJECTIVE)
    channel = NikonChannel(0, "BF", (255, 255, 255), None, None, modalities=("brightfield", "spinningDiskConfocal"))
    graph = I.acquisition_of(scope).channel_lightpath(channel, "Sample")
    check_ports(graph)
    assert chain(graph) == ["source", "sample", "objective", "camera"]
    assert elements(graph)["source"].label == "Transmitted light (implied)"


def test_a_widefield_channel_names_its_filter_only_when_nis_did():
    scope = NikonScope(modalities=("fluorescence", "camera"), **OBJECTIVE)
    plain = NikonChannel(0, "DAPI", None, 453.5, None, modalities=("fluorescence", "camera"))
    graph = I.acquisition_of(scope).channel_lightpath(plain, "Sample")
    check_ports(graph)
    assert chain(graph) == ["source", "objective", "sample", "camera"]
    named = NikonChannel(0, "DAPI", None, 453.5, None, modalities=("fluorescence",), filter_name="DAPI cube")
    graph = I.acquisition_of(scope).channel_lightpath(named, "Sample")
    check_ports(graph)
    assert chain(graph) == ["source", "filter", "objective", "sample", "camera"]


def test_the_microscope_device_names_the_model():
    scope = NikonScope(microscope="Nikon Ti2", modalities=("fluorescence", "camera"), camera=NikonCamera(name="Flash4.0"), **OBJECTIVE)
    found = {d.label: {s.name: s for s in d.settings} for d in I.acquisition_of(scope).devices()}
    assert found["microscope"]["instrument-model"].text == "nikon.widefield"
    assert found["microscope"]["model"].text == "Nikon Ti2"
    assert found["objective"]["immersion"].text == "OIL"
    assert found["camera"]["name"].text == "Flash4.0"


def test_a_scan_head_is_no_camera():
    scope = NikonScope(modalities=("laserScanConfocal",), camera=NikonCamera(name="Nikon A1plus"), zoom=2.5)
    labels = {d.label for d in I.acquisition_of(scope).devices()}
    assert "scan-head" in labels and "camera" not in labels
    channel = I.acquisition_of(scope).channel_devices(a1_channel())
    by_label = {d.label: {s.name for s in d.settings} for d in channel.devices}
    assert "frame-time" in by_label["scan-0"]
    assert by_label["detector-0"] == {"name", "high-voltage", "offset", "laser-power"}
    assert by_label["pinhole-0"] == {"diameter"}


def test_a_widefield_channel_has_no_pinhole():
    scope = NikonScope(modalities=("fluorescence", "camera"))
    channel = NikonChannel(0, "x", None, None, None, modalities=("fluorescence",), pinhole_um=50.0, camera=NikonCamera(exposure_ms=40.0))
    labels = {d.label for d in I.acquisition_of(scope).channel_devices(channel).devices}
    assert "pinhole-0" not in labels and "camera-0" in labels
