# nd2-converter

Converts Nikon ND2 files into mikro datasets **natively**: no Bio-Formats, no JVM, no
OME in between. One pyramided `ArrayDataset` per stage position.

It is the sibling of `lif-converter` and `ome-converter`, with the same output contract,
and reads the file with [nd2](https://pypi.org/project/nd2/):

- **Lazy reads.** Each frame is read out of the file when a chunk of it is uploaded,
  one frame per chunk. Nothing is decoded up front, and the file is never copied to a
  scratch Zarr. Only pyramid level 1 (¼ of the base, ⅛ for a z-stack) is persisted,
  next to the download, so that the coarser levels do not each re-read the base.
- **NIS-Elements' own metadata.** nd2's typed record of the run (attributes, text info,
  the experiment's loops, per-channel metadata) goes on the image anchor verbatim, as
  `{"format": "nikon-nd2", …}` in the anchor's metadata blob.
  - The image anchor pins no axis, so it holds over the whole image. It carries the
    stand, the objective (NA, immersion, order code), the camera (or scan head) and its
    binning, the stage, and the acquisition window, from the point's first frame to its
    last (UTC).
  - A time series also states each timepoint's own window on its `{t: i}` anchor, so
    irregular intervals survive. mikro derives the dataset's
    `acquisitionStart`/`acquisitionEnd` from these windows.
  - Each channel anchor gets its modality flags, its NIS colour, its emission and
    excitation, and the filter path's laser lines and detection band. On a camera system
    it also gets the exposure. On a point scanner it gets the PMT/GaAsP voltage, offset
    and laser power, plus the pinhole. These are recorded as device settings and a light
    path.
  - `nd2_metadata.py` documents where each of these lives. The camera, filter and
    detector details come from nd2's *unstructured* metadata, which nd2 calls
    experimental, so they degrade to absent rather than fail.
- **NIS-derived display.** Each channel is drawn in the colour NIS displayed it in, or,
  failing that, in the colour its emission implies. It opens on p1/p99 of what its layer
  shows: NIS keeps no display window this converter reads.
- **Tuned pyramids.** `levels=0` (the default) halves until the coarsest level is
  ≤ 512 px.

## Instrument models

Each file is laid out on the kind of microscope its channels' modality flags describe
(`nd2_instruments.py`). There is no system name to match, as there is in a LIF.

| Model | Flags | Light path |
|---|---|---|
| `nikon.point-scanning` | laserScanConfocal, pmt, gaasp, multiphoton, nonDescannedDetector | lines → combiner → scanner → objective → sample → objective → scanner → pinhole → band → PMT/GaAsP (A1, AX, C2). A non-descanned detector takes the light off the objective; the transmitted detector sits behind the sample. |
| `nikon.spinning-disk` | spinningDiskConfocal, sora, dsdConfocal, sweptField…, RCM, iSIM, liveSR | lines → disk (with its pinhole size) → objective → sample → objective → disk → camera |
| `nikon.sim` | SIM | lines → structured illumination → objective → sample → objective → camera |
| `nikon.tirf` | TIRF | lines → objective → sample → objective → camera |
| `nikon.widefield` | camera, fluorescence, brightfield, DIC, phase | light source → [named filter] → objective → sample → objective → [filter] → camera |
| `nikon.generic` | none | what the file names |

The first matching row wins. A channel whose own flags say transmitted light
(brightfield, DIC, phase) is drawn as lamp → sample → objective → camera on any model.

An element the file names is recorded: a laser line, the camera, the detector, the
pinhole, a named filter. One that only the design implies is labelled "(implied)" and
carries no numbers: the scanner, the disk without its pinhole size, the combiner, a
widefield light source. mikro has no instrument entity yet, so the model rides on the
image anchor's `microscope` device as `instrument-model`, `family` and `modalities`.

Only the files of the nd2 project's test set have been checked: Ti, Ti2, A1, spinning
disk, DS-Fi3/Qi2 colour cameras, Flash4.0, Kinetix, Andor. SIM and TIRF are modelled
from the flags alone.

## Placement

Every calibrated point is registered into one micrometer world. An axis NIS did not
calibrate (`axesCalibrated`) is not given a scale. `voxel_size()` answers 1.0 for such
an axis, which would claim one pixel is one micrometer. An uncalibrated image is staged
in its own pixel grid.

- **x/y**: NIS records each point's stage position as the centre of its field, read
  here from that point's first frame. The world is laid out the way the pixels are, so
  each stage centre is carried into it by the nearest axis-aligned orientation the file
  records. That is the pixel → stage matrix where NIS calibrated one, else the camera
  matrix, which on most Nikon cameras is a 180° turn. Without this, a grid of fields
  would be mirrored. The degree or two of rotation left over cannot be expressed by a
  scale-and-offset placement, so it is recorded on the image anchor's
  `camera-orientation` device and not applied.
- **z**: a stack is placed at its lowest plane, from the per-plane `Z Coord [µm]` NIS
  recorded. The frame metadata's stage z is not used: it is constant over a stack, and
  is the bottom plane in some files and the home plane in others. A stack NIS acquired
  top-down is turned over, so that z grows with the index.
- **All or nothing**: an image shares the world only where every spatial axis it has is
  placed. A stack whose planes were driven by a piezo NIS did not read out has no
  `Z Coord`, so it is registered, calibrated, into a micrometer space of its own. A
  stage at exactly (0, 0) is a system without an encoded stage, not a position.

## Multipoint runs and scenes

A multipoint run is one ND2 with a `P` axis. Every point is uploaded and registered.
Points whose fields touch share one mosaic scene: a grid NIS acquired but did not
stitch. Points that do not touch (the wells of a plate, or points picked across a
slide) each get a scene of their own, since a scene over them would be mostly empty
space. A mosaic is capped at 48 layers, so a larger touching grid also gets a scene per
point rather than none. Whether the points will share a scene is decided from the header
before any pixel is read, so each snapshot is rendered at the size of its scene. An
image NIS stitched itself is one field.

A colour camera with several channels (`C` × `S`, e.g. two DS-Fi3 exposures) becomes one
packed-RGB dataset per channel, and they share their point's scene.

## The action

**Convert Nikon ND2 file to Datasets** (`convert_nd2_file_to_datasets`) takes a mikro
`File`, plus `levels` (0 = auto), `method`, `stage_scene`, `world_name` and `snapshot`.
`snapshot` is on by default: each staged scene gets a maximum projection of its layers
as its snapshot. Each staged scene is also set as the default scene of every dataset in
it. It returns the datasets it made. If one point fails, it is logged and the rest still
convert. Legacy (JPEG 2000) ND2 files are refused.

## Developing

```sh
uv run pytest                              # offline: no server, no file
ND2_SAMPLES=~/nd2-samples uv run pytest    # also against nd2's sample files (see tests/test_samples.py)
uv run arkitekt check
uv run arkitekt plugin build -p linux/amd64
```
