"""Instrument metadata builders for the converters here: settings and device states an ingest records.

A converter reads a pile of loosely-typed header fields and has to turn them into
`SettingInput`s and `DeviceStateInput`s. The server enforces that a `SettingInput` fills
exactly one of its four value slots and that a device carries at least one setting; what
these helpers add is the convenience an absent header field needs: `setting` and `device`
return ``None`` when there is nothing to say, and `settings` / `devices` drop the ``None``s,
so no device ends up asserting ``model = "None"``.

Used to be ``mikro.metadata``; it is converter tooling, not client API, so it lives here.

    devices(
        device("objective", "OBJECTIVE",
               setting("magnification", number=60.0),
               setting("immersion", string="Oil")),
        device("camera", "DETECTOR", setting("exposure", quantity="50 ms")),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from kanne.scalars import GenericQuantity

if TYPE_CHECKING:
    from mikro.api.schema import DeviceStateInput, SettingInput

__all__ = ["device", "devices", "setting", "settings", "text"]


def text(value: object) -> str | None:
    """A None-safe string, unwrapping enums and decoding fixed-width byte fields.

    An enum serialises as ``"UnitsLength.MICROMETER"`` rather than ``"µm"``; a NUL-padded char
    field out of a binary header yields the text followed by sixty NULs; an empty or
    whitespace-only string becomes a setting that says nothing. All three are quiet, so all
    three are handled here.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        value = value.item()
    if isinstance(value, bytes):
        # Fixed-width char fields are NUL-padded; truncating at the first NUL is what the format means.
        value = value.split(b"\x00")[0].decode("latin-1", "replace")
    rendered = str(getattr(value, "value", value)).strip()
    return rendered or None


def setting(
    name: str,
    *,
    quantity: str | GenericQuantity | None = None,
    number: float | None = None,
    string: str | None = None,
    flag: bool | None = None,
) -> SettingInput | None:
    """One named value, or ``None`` when there is no value to name.

    Raises:
        ValueError: If more than one value slot is filled (the server refuses it too).
    """
    from mikro.api.schema import SettingInput

    filled = [v for v in (quantity, number, string, flag) if v is not None]
    if len(filled) > 1:
        raise ValueError(f"setting {name!r} fills {len(filled)} value slots, not 1: a SettingInput carries exactly one of quantity/number/string/flag")
    if not filled:
        return None
    return SettingInput(
        name=name,
        # A `GenericQuantity` is a pint-backed scalar, not a plain string; the wire rejects the bare `"50 ms"`.
        quantity=None if quantity is None else GenericQuantity(quantity),
        number=number,
        text=string,
        flag=flag,
    )


def settings(*items: SettingInput | None) -> list[SettingInput]:
    """The settings that actually carry a value."""
    return [item for item in items if item is not None]


def device(label: str, kind: str, *items: SettingInput | None) -> DeviceStateInput | None:
    """A device state, or ``None`` when the file said nothing about it."""
    from mikro.api.schema import DeviceStateInput

    kept = settings(*items)
    if not kept:
        return None
    return DeviceStateInput(label=label, kind=kind, settings=tuple(kept))


def devices(*items: DeviceStateInput | None) -> list[DeviceStateInput]:
    """The device states that actually say something."""
    return [item for item in items if item is not None]
