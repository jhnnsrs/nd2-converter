"""Offline checks of what is read out of NIS's records.

The ``SLxPictureMetadata`` fixtures are shaped like the trees nd2 hands back for the
sample files (``cluster.nd2``, an A1; ``t3p3z5c3.nd2``, a Ti2 with a Flash4.0): keys as
nd2 strips them, values as NIS wrote them.
"""

import datetime
import json
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from mikro.api.schema import ObjectiveImmersion

import nd2_metadata as N

A1_PLANE = {
    "FilterPath": {
        "Filter": {
            "i0000000000": {
                "UserName": "???",
                "Name": "???",
                "ExcitationSpectrum": {
                    "Point": {
                        "Point0": {"Type": 4, "Wavelength": 488.0},
                        "Point1": {"Type": 4, "Wavelength": 561.0},
                    }
                },
                "EmissionSpectrum": {
                    "Point": {"Point0": {"Type": 4, "Wavelength": 525.0, "TValue": 25.0}}
                },
            }
        }
    },
    "PinholeDiameter": 38.3,
}

A1_SETTING = {
    "CameraSetting": {
        "CameraUserName": "Nikon A1plus",
        "CameraFamilyName": "Nikon A1",
        "PropertiesQuality": {
            "CH1ChannelDyeName": "POPO-1/DNA",
            "CH2ChannelDyeName": "5-FAM/pH 9.0",
            "CH2PMTHighVoltage": 37,
            "CH2PMTOffset": -29,
            "CH2LaserPower": 0.5,
        },
    },
    "DeviceSetting": {"MicroscopePhysFullName": "Nikon Ti"},
}

CAMERA_SETTING = {
    "CameraSetting": {
        "CameraUserName": "Flash4.0, 9Z0021",
        "CameraFamilyName": "ecmC11440_22C",
        "FormatQuality": {"Desc": {"BinningX": 2.0, "BinningY": 2.0, "FormatText": "16-bit - Binning 2x2"}},
        "PropertiesQuality": {"Exposure": 40.0},
    },
    "ExposureTime": 40.0,
    "DeviceSetting": {"MicroscopePhysFullName": "1", "MicroscopeFullName": "Ti2 Microscope"},
    "ObjectiveSetting": {"ObjectiveName": "Plan Apo λ 10x", "ObjectiveCode": "MRD00105"},
}


def test_an_a1_filter_path_names_its_lines_and_band():
    name, lines, band = N._filter_path(A1_PLANE)
    assert name is None  # "???" names nothing
    assert lines == (488.0, 561.0)
    assert band == (500.0, 550.0)


def test_an_a1_detector_is_found_by_its_dye():
    detectors = N._scan_detectors(A1_SETTING)
    # CH1 names a dye but has no voltage: it did not image.
    assert set(detectors) == {"5-FAM/pH 9.0"}
    found = detectors["5-FAM/pH 9.0"]
    assert (found.name, found.high_voltage, found.offset, found.laser_power) == ("CH2", 37.0, -29.0, 0.5)


def test_a_camera_setting():
    camera = N._camera(CAMERA_SETTING)
    assert camera.name == "Flash4.0, 9Z0021"
    assert camera.exposure_ms == 40.0
    assert camera.binning == "2x2"


def test_a_bare_number_names_no_stand():
    assert N._stand_name("1") is None
    assert N._stand_name("Nikon Ti2") == "Nikon Ti2"


def test_nis_arrays_keep_their_order():
    assert N._items({"Point10": 10, "Point2": 2, "Point0": 0}) == [0, 2, 10]


def test_a_zero_stage_is_no_stage():
    assert N.stage_xy(0.0, 0.0) == (None, None)
    assert N.stage_xy(0.0, 12.5) == (0.0, 12.5)


def test_immersion_from_the_name_then_the_index():
    assert NikonScope(objective_name="Plan Apo 100x Oil DIC H").immersion == ObjectiveImmersion.OIL
    assert NikonScope(objective_name="Plan Apo Lambda S 40XC Sil").immersion == ObjectiveImmersion.OTHER
    assert NikonScope(objective_name="CFI Apo LWD 40x WI").immersion == ObjectiveImmersion.WATER
    assert NikonScope(objective_name="Plan Apo λ 20x", refractive_index=1.0).immersion == ObjectiveImmersion.AIR
    assert NikonScope(objective_name="Plan Apo λ 20x").immersion is None


NikonScope = N.NikonScope


def test_the_camera_rotation():
    assert round(NikonScope(camera_matrix=(-0.99966, -0.02623, 0.02623, -0.99966)).camera_rotation_deg, 1) == -178.5
    assert NikonScope().camera_rotation_deg is None


def test_frame_stamps_are_utc():
    """``top_to_bottom.nd2``'s first frame, stamped 18:00 local (CEST) by its text info."""
    stamp = N.jdn_to_datetime64(2461322.16689232)
    assert stamp.astype(datetime.datetime).replace(microsecond=0) == datetime.datetime(2026, 10, 8, 16, 0, 19)


def test_the_frame_table_anchors_relative_times_on_the_first_stamp():
    first = SimpleNamespace(
        channels=[SimpleNamespace(time=SimpleNamespace(absoluteJulianDayNumber=2461322.0, relativeTimeMs=1000.0))]
    )
    f = SimpleNamespace(
        loop_indices=[{"Z": 0}, {"Z": 1}],
        events=lambda orient: {
            "Index": [0, 1],
            "Time [s]": np.array([1.0, 3.5]),
            "Z Coord [µm]": np.array([10.0, 10.5]),
        },
        frame_metadata=lambda seq: first,
    )
    table = N.read_frame_table(f)
    assert table.z_um.tolist() == [10.0, 10.5]
    assert (table.times[1] - table.times[0]) == np.timedelta64(2_500_000, "us")
    assert table.times[0] == N.jdn_to_datetime64(2461322.0)


def test_a_frame_table_without_events_falls_back_to_each_frames_stamp():
    def frame(seq):
        return SimpleNamespace(
            channels=[SimpleNamespace(time=SimpleNamespace(absoluteJulianDayNumber=2461322.0 + seq / 86400, relativeTimeMs=0.0))]
        )

    def no_events(orient):
        raise RuntimeError("no events")

    f = SimpleNamespace(loop_indices=[{"T": 0}, {"T": 1}], events=no_events, frame_metadata=frame)
    table = N.read_frame_table(f)
    assert round((table.times[1] - table.times[0]) / np.timedelta64(1, "s")) == 1
    assert np.isnan(table.z_um).all()


def test_the_record_is_json_whatever_nd2_hands_back():
    f = SimpleNamespace(
        attributes=namedtuple("Attributes", "widthPx")(32),
        text_info={"date": "9/28/2021"},
        experiment=[N.ZStack(5, 1.0, True, 2, "Ti2 ZDrive")],
        metadata={"value": np.float32(1.5), "nan": float("nan")},
    )
    record = json.loads(N.image_record(f, "x.nd2", "7", N.Position(1, "A01", 1.0, 2.0, 3.0)))
    assert record["format"] == "nikon-nd2"
    assert record["position_name"] == "A01"
    assert record["experiment"][0]["device"] == "Ti2 ZDrive"
    assert record["metadata"] == {"value": 1.5, "nan": None}
    assert record["attributes"] == {"widthPx": 32}
