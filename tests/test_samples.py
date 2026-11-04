"""Checks against real ND2 files: the nd2 project's own test samples.

Skipped unless ``ND2_SAMPLES`` names a directory holding them. They are the assets of
https://github.com/tlambert03/nd2/releases/tag/test-data, e.g.::

    mkdir -p ~/nd2-samples && cd ~/nd2-samples
    for f in dims_rgb_c2x64y64 top_to_bottom bottom_to_top cluster t3p3z5c3 \\
             wellplate96_4_wells_without_jobs dims_p4z5t3c2y32x32 10ms_2xbin_100xmag; do
        curl -sfLO https://github.com/tlambert03/nd2/releases/download/test-data/$f.nd2
    done
    ND2_SAMPLES=~/nd2-samples uv run pytest

Every file found there is converted end to end through the action itself, with a
recording Mikro client and a fake world: the pixels are read, the pyramid is built and
level 1 persisted, the histograms, anchors, light paths and snapshots made.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app as M
import nd2_metadata as N

SAMPLES = Path(os.environ.get("ND2_SAMPLES", "/nonexistent"))
FILES = sorted(SAMPLES.glob("*.nd2")) if SAMPLES.is_dir() else []

pytestmark = [
    pytest.mark.samples,
    pytest.mark.skipif(not FILES, reason="ND2_SAMPLES does not name a directory of ND2 files"),
]


def sample(name):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not among the samples")
    import nd2

    return nd2.ND2File(path)


class Space:
    def __init__(self, name, log):
        self.id = self.name = name
        self.log = log

    def register(self, dataset, **kw):
        self.log.append((self.name, dataset.name, kw))


class Dataset:
    def __init__(self, name, data):
        self.name = name
        self.axis_names = list(data.dims)
        self.intrinsic_system = SimpleNamespace(id=f"{name}-intrinsic", name=name)

    def lens(self, **windows):
        return SimpleNamespace(id=f"{self.name}{windows}")


class Recorder:
    def __init__(self):
        self.calls = []

    def create_array_dataset(self, data, scales, name, axes, anchors, source_files):
        np.asarray(data)  # the read the upload makes
        for anchor in anchors:
            if anchor.ome_metadata is not None:
                json.loads(anchor.ome_metadata.metadata_string)
        self.calls.append(("create_array_dataset", name))
        return Dataset(name, data)

    def create_scene(self, name, coordinate_system, **kw):
        self.calls.append(("create_scene", name))
        return SimpleNamespace(
            id=name, name=name, world_coordinate_system=SimpleNamespace(name=str(coordinate_system))
        )

    def __getattr__(self, name):
        return lambda **kw: self.calls.append((name, kw))


def convert(path, monkeypatch, tmp_path):
    log = []
    monkeypatch.setattr(M, "space_3d", lambda mikro, name, unit="micrometer": Space(name, log))
    monkeypatch.setenv("MIKRO_SCRATCH_DIR", str(tmp_path))
    mikro = Recorder()
    file = SimpleNamespace(
        name=path.name,
        id="1",
        store=SimpleNamespace(download=lambda target: shutil.copy(path, target)),
    )
    datasets = M.convert_nd2_file_to_datasets(file, mikro, SimpleNamespace(progress=lambda *a: None))
    return datasets, mikro.calls, log


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_every_sample_converts(path, monkeypatch, tmp_path):
    datasets, calls, _ = convert(path, monkeypatch, tmp_path)
    assert datasets
    assert any(name == "create_scene" for name, _ in calls)
    assert any(name == "create_scene_snapshot" for name, _ in calls)
    # The scratch directory is gone with the run.
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["top_to_bottom.nd2", "bottom_to_top.nd2"])
def test_a_stack_is_placed_at_its_lowest_plane_and_grows_upwards(name, monkeypatch, tmp_path):
    with sample(name) as f:  # skips when the file is not among the samples
        z = N.read_frame_table(f).z_um
    _, _, log = convert(SAMPLES / name, monkeypatch, tmp_path)
    [(_, _, kw)] = log
    assert kw["z"] == pytest.approx(z.min())
    assert kw["scale"]["z"] == pytest.approx(0.3)


def test_a_top_down_stack_is_turned_over():
    with sample("top_to_bottom.nd2") as f:
        stack = N.z_stack(f)
        assert not stack.bottom_to_top
        whole = M.file_array(f)
        up, _, _ = M.field_array(whole, None, None, True)
        np.testing.assert_array_equal(np.asarray(up.isel(c=0, z=0)), np.asarray(whole.isel(Z=-1)))


def test_wells_are_laid_out_as_on_the_plate(monkeypatch, tmp_path):
    """A01 → A04 runs left to right, A → B → C top to bottom, as the plate does."""
    path = SAMPLES / "wellplate96_4_wells_without_jobs.nd2"
    if not path.exists():
        pytest.skip("wellplate96_4_wells_without_jobs.nd2 is not among the samples")
    _, _, log = convert(path, monkeypatch, tmp_path)
    at = {dataset.rsplit(" - ", 1)[1]: kw for _, dataset, kw in log}
    assert at["A01"]["x"] < at["C02"]["x"] < at["A04"]["x"]
    assert at["A04"]["x"] == pytest.approx(at["B04"]["x"])
    assert at["A01"]["y"] == pytest.approx(at["A04"]["y"])
    assert at["A04"]["y"] < at["B04"]["y"] < at["C02"]["y"]


def test_a_colour_camera_with_two_channels_is_two_pictures_in_one_scene(monkeypatch, tmp_path):
    path = SAMPLES / "dims_rgb_c2x64y64.nd2"
    if not path.exists():
        pytest.skip("dims_rgb_c2x64y64.nd2 is not among the samples")
    datasets, calls, _ = convert(path, monkeypatch, tmp_path)
    assert len(datasets) == 2
    assert [name for name, _ in calls].count("create_rgb_layer") == 2
    assert [name for name, _ in calls].count("create_scene") == 1


def test_a_placeholder_stage_is_not_placed(monkeypatch, tmp_path):
    """The Ti2 simulator puts its first point at exactly (0, 0)."""
    with sample("t3p3z5c3.nd2") as f:
        positions = N.read_positions(f)
    assert (positions[0].x_um, positions[0].y_um) == (None, None)
    assert positions[1].x_um is not None


def test_frame_times_are_each_frames_own_stamp():
    with sample("dims_p4z5t3c2y32x32.nd2") as f:
        table = N.read_frame_table(f)
        own = [
            N.jdn_to_datetime64(f.frame_metadata(i).channels[0].time.absoluteJulianDayNumber)
            for i in range(len(table.times))
        ]
    deviation = np.abs((table.times - np.array(own)) / np.timedelta64(1, "ms"))
    assert deviation.max() < 1.0


def test_an_a1_channel_reads_its_detector():
    with sample("cluster.nd2") as f:
        channels = N.read_channels(f)
        scope = N.read_scope(f)
    assert {c.detector.name for c in channels} == {"CH2", "CH4"}
    assert channels[0].emission_band == (500.0, 550.0)
    assert channels[0].excitation_lines == (488.0, 561.0)
    from nd2_instruments import identify

    assert identify(scope).key == "nikon.point-scanning"
