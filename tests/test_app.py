"""Offline checks of the converter: what it reads, where it places it, what it authors.

No backend and no ND2 file: nd2's array is a stand-in xarray, its metadata stand-in
records, and the Mikro client a recorder, so the assertions land on the calls that would
reach the server rather than on a property that predicts them. The checks against real
files are in `test_samples.py`.

The layer, pyramid, preview and staging checks are the LIF converter's, unchanged: the
converters share that code, and the rule it serves.

    ONE LAYER PER CHANNEL. The only exception is a packed RGB(A) pixel.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import app as M
from nd2_metadata import Calibration, FrameTable, NikonChannel, NikonScope, Position, ZStack

class _RecordingMikro:
    """Stands in for the Mikro client: records every mutation the layer author makes."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        return lambda **kw: self._calls.append((name, kw))


@pytest.mark.parametrize(
    ("sizes", "levels"),
    [
        ({"y": 512, "x": 512}, 2),
        ({"y": 1024, "x": 1024}, 2),
        ({"y": 11003, "x": 11022}, 6),
        ({"y": 64, "x": 64}, 2),
        ({"y": 3741, "x": 3680}, 4),
    ],
)
def test_auto_levels(sizes, levels):
    assert M.auto_levels(sizes) == levels

@pytest.mark.parametrize(
    ("packed", "channels", "dtype", "expected"),
    [
        (True, 3, np.uint8, True),
        (True, 4, np.uint8, True),
        (False, 3, np.uint8, False),
        (True, 6, np.uint8, False),
        # An RGB layer is drawn unstretched over 0..255: a deeper sample axis is a colour
        # camera's acquisition and gets a layer and a window per sample instead.
        (True, 3, np.uint16, False),
        (True, 3, np.float32, False),
    ],
)
def test_packs_one_pixel(packed, channels, dtype, expected):
    assert M.packs_one_pixel(packed, channels, dtype) is expected

def test_display_window():
    assert M.display_window(np.arange(101.0).reshape(1, 101)) == (1.0, 99.0)
    # Over 99 % one value: the percentiles coincide, and a window of no width draws
    # nothing.
    dots = np.zeros((64, 64), dtype=np.uint16)
    dots[0, :8] = 500
    assert M.display_window(dots) == (0.0, 500.0)
    assert M.display_window(np.full((4, 4), 7.0)) == (7.0, 8.0)



class FakeId:
    id = "fake"
    name = "fake"


class FakeLens:
    def __init__(self, id):
        self.id = id


class FakeDataset:
    """A dataset as far as the layer author reads one: its axes, and its lenses.

    A lens' id spells out what it pins — ``"lens"`` for the whole lens, ``"lens:c=2"``
    for channel 2 — so a recorded call can be asked which channel it draws.
    """

    axis_names = ("c", "z", "y", "x")

    def lens(self, **selections):
        for axis in selections:
            if axis not in self.axis_names:
                raise ValueError(f"Invalid axis {axis!r} for axes {list(self.axis_names)}")
        if not selections:
            return FakeLens("lens")
        return FakeLens("lens:" + ",".join(f"{a}={v}" for a, v in selections.items()))


def pinned_channel(lens_id):
    """The channel a fake lens id pins, or None for the whole lens and for no lens."""
    if not isinstance(lens_id, str) or not lens_id.startswith("lens:"):
        return None
    pins = dict(pin.split("=") for pin in lens_id[len("lens:") :].split(","))
    return int(pins["c"]) if "c" in pins else None


def fake_specs(n):
    return [
        M.ChannelSpec(
            index=i,
            label=f"ch{i}",
            color=(i * 60 % 256, 0, 0, 255),
            histogram=M.ValueHistogramInput(
                histogram=[1.0], bins=[0.0, 1.0], min=0.0, max=1.0, p1=0.0, p99=1.0
            ),
            # Deliberately not the histogram's p1/p99, so a check can tell which of the
            # two a layer was given.
            clim=(10.0 + i, 20.0 + i),
            nikon=None,
        )
        for i in range(n)
    ]


def layers_for(specs, implicit_rgb, is_volume=True):
    calls = []
    made = M.author_channel_layers(
        _RecordingMikro(calls), FakeDataset(), FakeId(), specs, is_volume, implicit_rgb
    )
    assert made == len(calls), "layer count must match the calls made"
    return calls


def selected_channels(call):
    """Every channel index this layer covers, whichever kind of layer it is.

    An unrecognised call is an error rather than an empty list: otherwise the converter
    moving to a mutation this walker does not know would make every check below pass
    vacuously against nothing at all.
    """
    name, kw = call
    if name in ("create_intensity_layer", "create_volume_layer"):
        index = pinned_channel(kw.get("lens"))
        assert index is not None, f"{name} call's lens pins no channel: {kw!r}"
        return [index]
    if name == "create_rgb_layer":
        found = [pinned_channel(kw.get(k)) for k in ("red_lens", "green_lens", "blue_lens")]
        assert None not in found, f"{name} call is missing a component lens: {kw!r}"
        return found
    if name == "create_layer":
        found = []

        def walk(node):
            if node is None:
                return
            if node.get("kind") == "channel":
                index = pinned_channel(node.get("lens"))
                assert index is not None, f"channel node's lens pins no channel: {node!r}"
                found.append(index)
            for child in node.get("children") or ():
                walk(child)

        graph = kw.get("render_graph")
        assert graph is not None, f"create_layer call carries no render_graph: {kw!r}"
        walk(graph.model_dump()["root"])
        assert found, f"create_layer graph has no channel node: {kw!r}"
        return found
    raise AssertionError(f"unrecognised layer mutation {name!r}; teach the walker about it")


def test_a_z_stack_is_one_volume_layer_per_channel():
    four = layers_for(fake_specs(4), False)
    assert len(four) == 4
    assert {n for n, _ in four} == {"create_volume_layer"}
    assert sorted(i for c in four for i in selected_channels(c)) == [0, 1, 2, 3]
    assert sorted(tuple(kw["color"]) for _, kw in four) == sorted(
        tuple(s.color) for s in fake_specs(4)
    )
    assert sorted((kw["clim_min"], kw["clim_max"]) for _, kw in four) == [
        (10.0 + i, 20.0 + i) for i in range(4)
    ]


def test_a_packed_rgb_z_stack_is_one_render_graph_layer():
    rgb = layers_for(fake_specs(3), True)
    assert len(rgb) == 1
    assert rgb[0][0] == "create_layer"
    assert sorted(selected_channels(rgb[0])) == [0, 1, 2]
    assert len(layers_for(fake_specs(4), True)) == 1


@pytest.mark.parametrize("n", [2, 5])
def test_implicit_rgb_over_a_non_pixel_channel_count_is_refused(n):
    assert len(layers_for(fake_specs(n), True)) == n


def test_flat_channels_are_intensity_layers():
    flat = layers_for(fake_specs(4), False, is_volume=False)
    assert {n for n, _ in flat} == {"create_intensity_layer"}
    assert sorted(i for c in flat for i in selected_channels(c)) == [0, 1, 2, 3]
    assert [kw.get("mode") for _, kw in flat] == [None] * 4


def test_a_flat_packed_rgb_pixel_is_one_rgb_layer():
    flat_rgb = layers_for(fake_specs(3), True, is_volume=False)
    assert [n for n, _ in flat_rgb] == ["create_rgb_layer"]
    assert selected_channels(flat_rgb[0]) == [0, 1, 2]
    kw = flat_rgb[0][1]
    assert (kw["clim_min"], kw["clim_max"]) == (0.0, 255.0)
    assert "blending" not in kw


def test_rgba_labels_and_colours():
    assert [M.channel_label(None, i, True) for i in range(4)] == [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]
    assert [M.channel_color(None, i, True, 4) for i in range(4)] == [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 255, 255),
    ]


def test_a_persisted_level_one_changes_no_value(tmp_path):
    import dask.array as da

    base = xr.DataArray(
        da.from_array(np.random.default_rng(0).integers(0, 255, (2, 4, 64, 64), dtype=np.uint8), chunks=(1, 1, 64, 64)),
        dims=("c", "z", "y", "x"),
    )
    lazy = M.build_pyramid(base, levels=4, method="max", materialize=False)
    persisted = M.image_pyramid(base, 4, "max", {}, str(tmp_path))
    assert len(persisted) == 4
    for a, b in zip(lazy, persisted):
        assert a.dims == b.dims and a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Level 1 is read back from the scratch directory, not re-derived from the base.
    assert any(tmp_path.iterdir())


# --- the scene snapshot ---------------------------------------------------------------


def test_a_preview_is_the_maximum_projection_in_each_channels_window():
    data = np.zeros((2, 3, 4, 4), dtype=np.uint16)
    data[0, 1, 0, 0] = 20  # channel 0, bright at one plane only
    data[1, 2, 3, 3] = 15  # channel 1, halfway into its window
    level = xr.DataArray(data, dims=("c", "z", "y", "x"))
    specs = [
        M.ChannelSpec(0, "a", (255, 0, 0, 255), None, (0.0, 20.0), None),
        M.ChannelSpec(1, "b", (0, 255, 0, 255), None, (10.0, 20.0), None),
    ]

    rgb = M.render_preview(level, specs, implicit_rgb=False, side=4)

    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0)
    assert tuple(rgb[3, 3]) == (0, 127, 0)
    assert rgb[1:3, 1:3].max() == 0


def test_a_preview_fits_its_side():
    level = xr.DataArray(np.zeros((1, 100, 50), dtype=np.uint8), dims=("c", "y", "x"))
    specs = [M.ChannelSpec(0, "a", (255, 255, 255, 255), None, (0.0, 1.0), None)]
    assert M.render_preview(level, specs, False, side=32).shape == (25, 13, 3)


def test_an_rgba_previews_alpha_is_not_drawn():
    level = xr.DataArray(np.full((4, 2, 2), 255, dtype=np.uint8), dims=("c", "y", "x"))
    specs = [
        M.ChannelSpec(i, "", M.channel_color(None, i, True, 4), None, (0.0, 255.0), None)
        for i in range(4)
    ]
    assert tuple(M.render_preview(level, specs, True, side=2)[0, 0]) == (255, 255, 255)


def test_tiles_are_laid_out_at_their_stage_positions():
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 200
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[..., 1] = 100
    canvas = M.mosaic(
        [
            M.Preview(red, (0.0, 0.0), (1.0, 1.0)),
            M.Preview(green, (0.0, 2.0), (1.0, 1.0)),
        ]
    )
    assert canvas.shape == (2, 4, 3)
    assert tuple(canvas[0, 0]) == (200, 0, 0)
    assert tuple(canvas[1, 3]) == (0, 100, 0)


def test_the_snapshot_is_a_png():
    import zlib

    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    png = M.encode_png(rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8 : start - 4], "big")
    rows = np.frombuffer(zlib.decompress(png[start : start + length]), np.uint8)
    rows = rows.reshape(2, 1 + 3 * 3)
    assert (rows[:, 0] == 0).all()
    assert (rows[:, 1:].reshape(2, 3, 3) == rgb).all()


def test_a_scene_gets_its_snapshot(tmp_path):
    calls = []
    scene = M.Scene.model_construct(id="s1", name="scene")
    entry = M.Converted(
        0, "x", None, [], True, False, False,
        M.Preview(np.zeros((2, 2, 3), np.uint8), None, None),
    )
    M.snapshot_scene(_RecordingMikro(calls), scene, [entry], str(tmp_path))
    [(name, kw)] = calls
    assert name == "create_scene_snapshot"
    assert kw["scene"] == "s1"
    assert kw["file"].name.endswith(".png")


def test_a_mosaic_never_exceeds_the_snapshot_side():
    tile = np.full((4, 4, 3), 255, dtype=np.uint8)
    far = 10 * M.SNAPSHOT_PIXELS
    canvas = M.mosaic(
        [M.Preview(tile, (0.0, 0.0), (1.0, 1.0)), M.Preview(tile, (0.0, far), (1.0, 1.0))]
    )
    assert max(canvas.shape[:2]) <= M.SNAPSHOT_PIXELS
    assert canvas[0, 0].any() and canvas[0, -1].any()


def test_an_unplaced_series_is_staged_alone_in_its_own_space(monkeypatch):
    composed = []
    monkeypatch.setattr(
        M, "compose_scene", lambda mikro, name, system, members: composed.append((name, system, [m.name for m in members]))
    )
    world, own = object(), object()
    entries = [
        M.Converted(0, "a", FakeDataset(), fake_specs(1), True, False, False, None),
        M.Converted(1, "b", FakeDataset(), fake_specs(1), True, False, False, None),
        M.Converted(2, "c", FakeDataset(), fake_specs(1), False, False, False, None, own),
    ]
    M.stage_group(None, "file", world, entries)
    assert composed == [("file", world, ["a", "b"]), ("c", own, ["c"])]


def test_a_composed_scene_is_each_datasets_default():
    calls = []

    class Mikro(_RecordingMikro):
        def create_scene(self, **kw):
            calls.append(("create_scene", kw))
            return M.Scene.model_construct(
                id="s1", name=kw["name"], world_coordinate_system=FakeId()
            )

    entries = [
        M.Converted(tile, f"t{tile}", FakeDataset(), fake_specs(1), True, False, False, None)
        for tile in range(2)
    ]
    M.compose_scene(Mikro(calls), "mosaic", FakeId(), entries)
    nominations = [kw for name, kw in calls if name == "set_lens_default_scene"]
    assert nominations == [{"lens": "lens", "scene": "s1"}] * 2


def test_the_app_declares_its_action():
    """One action, taking a mikro file and handed a mikro client."""
    assert M.app.services == ["mikro"]
    declaration = M.app.registry.to_implement_agent_input().model_dump(
        mode="json", exclude_none=True
    )
    [implementation] = declaration["implementations"]
    assert implementation["interface"] == "convert_nd2_file_to_datasets"
    args = {arg["key"]: arg for arg in implementation["definition"]["args"]}
    assert args["file"]["identifier"] == "@mikro/file"
    assert args["levels"]["default"] == 0
    # The client and the task are injected, not ports.
    assert "mikro" not in args and "task" not in args


# --- nd2's array, one point at a time -------------------------------------------------


def whole(dims, shape, fill=None):
    data = np.zeros(shape, dtype=np.uint8) if fill is None else fill
    return xr.DataArray(data, dims=dims)


@pytest.mark.parametrize(
    ("dims", "shape", "expected"),
    [
        # A colour camera: nd2's sample axis.
        (("Y", "X", "S"), (8, 8, 3), (3, True)),
        # Dyes, stored per channel.
        (("C", "Y", "X"), (3, 8, 8), (3, False)),
        (("Z", "C", "Y", "X"), (5, 4, 8, 8), (4, False)),
        # One channel, dropped by nd2: c comes back at extent 1.
        (("Y", "X"), (8, 8), (1, False)),
    ],
)
def test_the_sample_axis_decides_packing(dims, shape, expected):
    array, _, packed = M.field_array(whole(dims, shape), None, None, False)
    assert (int(array.sizes["c"]), packed) == expected


def test_several_colour_channels_are_converted_one_at_a_time():
    data = np.arange(2, dtype=np.uint8)[:, None, None, None] * np.ones((2, 4, 4, 3), np.uint8)
    both = whole(("C", "Y", "X", "S"), data.shape, data)
    with pytest.raises(ValueError):
        M.field_array(both, None, None, False)
    for c in range(2):
        array, _, packed = M.field_array(both, None, c, False)
        assert packed and array.sizes["c"] == 3 and int(array.max()) == c


def test_each_point_is_its_own_pixels():
    data = np.arange(3, dtype=np.uint8)[:, None, None, None] * np.ones((3, 2, 4, 4), np.uint8)
    run = whole(("P", "C", "Y", "X"), data.shape, data)
    for p in range(3):
        array, extra, _ = M.field_array(run, p, None, False)
        assert "p" not in array.dims and not extra
        assert int(array.max()) == p


def test_a_top_down_stack_is_turned_over():
    data = np.arange(4, dtype=np.uint8)[:, None, None] * np.ones((4, 2, 2), np.uint8)
    stack = whole(("Z", "Y", "X"), data.shape, data)
    up, _, _ = M.field_array(stack, None, None, True)
    assert [int(up.isel(c=0, z=i).max()) for i in range(4)] == [3, 2, 1, 0]
    kept, _, _ = M.field_array(stack, None, None, False)
    assert [int(kept.isel(c=0, z=i).max()) for i in range(4)] == [0, 1, 2, 3]


def test_a_custom_loop_is_declared_index_not_space():
    _, extra, _ = M.field_array(whole(("U", "C", "Y", "X"), (5, 2, 4, 4)), None, None, False)
    assert extra == {"U": "INDEX"}


# --- placement -----------------------------------------------------------------------


def test_voxel_scale():
    scale, note = M.voxel_scale(Calibration(0.65, 0.65, 0.3), ["c", "z", "y", "x"])
    assert scale == {"z": 0.3, "y": 0.65, "x": 0.65} and note is None
    scale, note = M.voxel_scale(Calibration(0.65, 0.65, None), ["c", "z", "y", "x"])
    assert scale["z"] == 1.0 and "z spacing unknown" in note


def test_uncalibrated_pixels_are_not_registered():
    scale, note = M.voxel_scale(Calibration(None, None, None), ["c", "y", "x"])
    assert scale is None and "not calibrated" in note


@pytest.mark.parametrize(
    ("scope", "column", "row"),
    [
        # The 180° most NIS cameras sit at (`10ms_2xbin_100xmag.nd2`): a column further
        # right is a stage x further left, a row further down a stage y further up.
        (NikonScope(camera_matrix=(-0.9997, -0.0262, 0.0262, -0.9997)), (0, -1), (1, 1)),
        # Upright (`gain1.nd2`): image y points up, rows go down the stage.
        (NikonScope(camera_matrix=(1.0, 0.0, 0.0, 1.0)), (0, 1), (1, -1)),
        # A calibrated pixel → stage matrix wins (`cluster.nd2`'s).
        (
            NikonScope(
                camera_matrix=(1.0, 0.0, 0.0, 1.0),
                pixel_to_stage=(-0.168, 0.0009, 3.17, 0.0009, 0.168, -2.62),
            ),
            (0, -1),
            (1, 1),
        ),
        # A camera turned a quarter: columns follow the stage's y, and since
        # stage_x = c·y_up = -y_up, a row down (y_up = -1) is a stage x further on.
        (NikonScope(camera_matrix=(0.0, 1.0, -1.0, 0.0)), (1, 1), (0, 1)),
    ],
)
def test_the_orientation_is_read_from_the_file(scope, column, row):
    found = M.orientation_of(scope)
    assert (found.column, found.row) == (column, row)


def test_the_camera_and_the_pixel_to_stage_matrix_agree():
    """Every sample that records both says the same thing about the axes."""
    pairs = [
        ((-0.99966, -0.02623, 0.02623, -0.99966), (-0.12896, -0.00338, 44.2, -0.00338, 0.12896, -31.9)),
        ((1.0, 0.0, 0.0, 1.0), (0.18337, 0.0, -220.0, -0.0, -0.18337, 220.0)),
        ((0.99930, -0.03734, 0.03734, 0.99930), (0.15828, -0.00591, -12.5, -0.00591, -0.15828, 12.8)),
        ((-0.99998, 0.00559, -0.00559, -0.99998), (-0.16781, 0.00094, 3.17, 0.00094, 0.16781, -2.62)),
    ]
    for camera, pixel_to_stage in pairs:
        by_camera = M.orientation_of(NikonScope(camera_matrix=camera))
        by_calibration = M.orientation_of(NikonScope(pixel_to_stage=pixel_to_stage))
        assert (by_camera.column, by_camera.row) == (by_calibration.column, by_calibration.row)


def test_no_orientation_takes_the_stage_axes_as_the_image():
    assert M.orientation_of(NikonScope()) is M.IDENTITY_ORIENTATION


def test_the_field_centre_sits_at_the_stage_centre():
    flipped = M.orientation_of(NikonScope(camera_matrix=(-1.0, 0.0, 0.0, -1.0)))
    centre = M.stage_centre(Position(0, "A01", 100.0, 50.0, None), flipped)
    assert (centre.x, centre.y) == (-100.0, 50.0)
    data = xr.DataArray(np.zeros((1, 11, 21)), dims=("c", "y", "x"))
    offsets = M.world_offsets(centre, {"y": 2.0, "x": 1.0}, data, None)
    assert offsets == {"x": -110.0, "y": 40.0}


def test_no_stage_centre_without_a_stage():
    assert M.stage_centre(Position(0, None, None, None, 5.0), M.IDENTITY_ORIENTATION) is None


def test_only_a_fully_placed_image_shares_the_world():
    centre = M.stage_centre(Position(None, None, 10.0, 10.0, None), M.IDENTITY_ORIENTATION)
    stack = xr.DataArray(np.zeros((1, 3, 4, 4)), dims=("c", "z", "y", "x"))
    scale = {"z": 1.0, "y": 1.0, "x": 1.0}
    # A stack whose planes NIS did not place along z is not in the world at all.
    assert M.world_offsets(centre, scale, stack, None) is None
    assert M.world_offsets(centre, scale, stack, 100.0)["z"] == 100.0
    assert M.world_offsets(None, scale, stack, 100.0) is None


def table(loops, z=None, times=None):
    return FrameTable.of_loops(loops, times, z)


def test_the_stack_bottom_is_its_lowest_recorded_plane():
    loops = [{"P": p, "Z": i} for p in range(2) for i in range(3)]
    frames = table(loops, z=[12.0, 11.0, 10.0, 22.0, 21.0, 20.0])
    stack = ZStack(3, 1.0, False, 1, "Ti2 ZDrive")
    assert M.stack_bottom(frames, 0, stack) == 10.0
    assert M.stack_bottom(frames, 1, stack) == 20.0


def test_a_stack_without_recorded_planes_has_no_bottom():
    """A piezo NIS did not read out: the frame's stage z is not a plane's."""
    loops = [{"Z": i} for i in range(3)]
    stack = ZStack(3, 0.5, True, 1, "Piezo")
    assert M.stack_bottom(table(loops), None, stack) is None
    assert M.stack_bottom(table(loops, z=[0.0, 0.0, 0.0]), None, stack) is None


def stamps(*seconds):
    return [np.datetime64("2026-10-08T16:00:00", "us") + np.timedelta64(int(s * 1e6), "us") for s in seconds]


def test_a_points_window_runs_from_its_first_frame_to_its_last():
    loops = [{"T": t, "P": p} for t in range(2) for p in range(2)]
    frames = table(loops, times=stamps(0, 1, 10, 11))
    window, per_t = M.frame_windows(frames, 1, 2)
    assert (window.start.second, window.end.second) == (1, 11)
    assert window.start.tzinfo is not None
    assert [(t, w.start.second) for t, w in per_t] == [(0, 1), (1, 11)]


def test_one_timepoint_states_no_timepoints():
    frames = table([{"Z": 0}, {"Z": 1}], times=stamps(0, 1))
    window, per_t = M.frame_windows(frames, None, 1)
    assert window is not None and per_t == []


def test_unstamped_frames_state_no_window():
    window, per_t = M.frame_windows(table([{"T": 0}, {"T": 1}]), None, 2)
    assert window is None and per_t == []


# --- which points share a scene -------------------------------------------------------


def converted(series, box):
    return M.Converted(series, f"p{series}", None, [], True, False, False, footprint=box, point_name=f"p{series}")


def test_a_grid_of_touching_fields_is_one_mosaic():
    grid = [[converted(i, (r * 95.0, c * 95.0, r * 95.0 + 100, c * 95.0 + 100))] for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
    assert M.share_a_scene(grid)


def test_wells_far_apart_are_not():
    wells = [[converted(i, (0.0, i * 9000.0, 100.0, i * 9000.0 + 100))] for i in range(3)]
    assert not M.share_a_scene(wells)


def test_a_field_set_apart_breaks_the_mosaic():
    fields = [
        [converted(0, (0.0, 0.0, 100.0, 100.0))],
        [converted(1, (0.0, 105.0, 100.0, 205.0))],
        [converted(2, (0.0, 1000.0, 100.0, 1100.0))],
    ]
    assert not M.share_a_scene(fields)


def test_an_unplaced_point_shares_no_mosaic():
    assert not M.share_a_scene([[converted(0, (0.0, 0.0, 1.0, 1.0))], [converted(1, None)]])


def test_the_channels_of_one_point_share_its_scene(monkeypatch):
    staged = []
    monkeypatch.setattr(
        M, "stage_group", lambda mikro, title, world, entries, scratch: staged.append((title, [e.name for e in entries]))
    )
    file = SimpleNamespace(name="x.nd2")
    a0 = M.Converted(0, "x - p1 - BF", None, [], True, False, True, footprint=(0.0, 0.0, 1.0, 1.0), point_name="x - p1")
    a1 = M.Converted(0, "x - p1 - DIC", None, [], True, False, True, footprint=(0.0, 0.0, 1.0, 1.0), point_name="x - p1")
    b0 = M.Converted(1, "x - p2 - BF", None, [], True, False, True, footprint=(0.0, 900.0, 1.0, 901.0), point_name="x - p2")
    M.stage_points(None, file, None, [a0, a1, b0], None)
    assert staged == [("x - p1", ["x - p1 - BF", "x - p1 - DIC"]), ("x - p2", ["x - p2 - BF"])]


def test_touching_points_are_staged_together(monkeypatch):
    staged = []
    monkeypatch.setattr(
        M, "stage_group", lambda mikro, title, world, entries, scratch: staged.append((title, len(entries)))
    )
    file = SimpleNamespace(name="x.nd2")
    entries = [converted(i, (0.0, i * 100.0, 100.0, i * 100.0 + 100)) for i in range(3)]
    M.stage_points(None, file, None, entries, None)
    assert staged == [("x.nd2 (3 points)", 3)]


# --- colours --------------------------------------------------------------------------


def channel(**kw):
    return NikonChannel(index=0, name="c", color=None, emission_nm=None, excitation_nm=None, **kw)


def test_colours_follow_nis_then_the_emission_then_the_palette():
    assert M.channel_color(
        NikonChannel(0, "GFP", (141, 255, 0), 535.0, 480.0), 0, False, 2
    ) == (141, 255, 0, 255)
    # Black is NIS's "no colour".
    by_emission = M.channel_color(NikonChannel(0, "x", (0, 0, 0), 690.0, None), 0, False, 2)
    assert by_emission[0] > 200 and by_emission[2] < 100
    banded = M.channel_color(channel(emission_band=(500.0, 550.0)), 0, False, 2)
    assert banded[1] > 200
    assert M.channel_color(channel(), 1, False, 2) == M.fallback_color(1, 2)


def test_a_channel_label_is_its_nis_name():
    assert M.channel_label(NikonChannel(0, "DAPI", None, None, None), 0, False) == "DAPI"
    assert M.channel_label(NikonChannel(1, None, None, None, None), 1, False) == "Channel 1"


def test_a_touching_grid_over_the_layer_budget_gets_a_scene_per_point(monkeypatch):
    """A 6×5 grid of two-channel fields is 60 layers: no mosaic, but no point unstaged."""
    staged = []
    monkeypatch.setattr(
        M, "stage_group", lambda mikro, title, world, entries, scratch: staged.append(title)
    )
    entries = [
        M.Converted(
            i,
            f"x - p{i}",
            None,
            fake_specs(2),
            True,
            False,
            False,
            footprint=(r * 95.0, c * 95.0, r * 95.0 + 100, c * 95.0 + 100),
            point_name=f"x - p{i}",
        )
        for i, (r, c) in enumerate((r, c) for r in range(6) for c in range(5))
    ]
    assert M.share_a_scene([[e] for e in entries])
    M.stage_points(None, SimpleNamespace(name="x.nd2"), None, entries, None)
    assert staged == [f"x - p{i}" for i in range(30)]


def header(points, channels=2, side=100):
    return SimpleNamespace(sizes={"P": points, "C": channels, "Y": side, "X": side})


def test_a_mosaic_is_planned_from_the_header():
    calibration = Calibration(1.0, 1.0, None)
    scope = NikonScope(camera_matrix=(1.0, 0.0, 0.0, 1.0))
    grid = [Position(i, None, i * 95.0, 0.0, None) for i in range(4)]
    assert M.planned_mosaic(header(4), grid, scope, calibration)
    wells = [Position(i, None, i * 9000.0, 0.0, None) for i in range(4)]
    assert not M.planned_mosaic(header(4), wells, scope, calibration)
    # Past the layer budget it will not be one either.
    long_grid = [Position(i, None, i * 95.0, 0.0, None) for i in range(30)]
    assert not M.planned_mosaic(header(30), long_grid, scope, calibration)
    # Nor without a calibration to size the fields by.
    assert not M.planned_mosaic(header(4), grid, scope, Calibration(None, None, None))


def test_a_frame_table_selects_at_scale():
    """96 wells × 500 timepoints × 3 planes: every point's windows in well under a second."""
    import time

    points, timepoints, planes = 96, 500, 3
    t, p, z = np.meshgrid(np.arange(timepoints), np.arange(points), np.arange(planes), indexing="ij")
    count = t.size
    frames = FrameTable(
        {"T": t.ravel(), "P": p.ravel(), "Z": z.ravel()},
        np.datetime64("2026-01-01T00:00:00", "us") + np.arange(count).astype("timedelta64[s]"),
        np.full(count, np.nan),
    )
    start = time.perf_counter()
    for point in range(points):
        whole, per_t = M.frame_windows(frames, point, timepoints)
    elapsed = time.perf_counter() - start
    assert len(per_t) == timepoints
    assert elapsed < 5.0, f"{elapsed:.1f} s"
