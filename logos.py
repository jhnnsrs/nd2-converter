"""The app's logo, and one logo per action

The app's logo is a LOGO blok (an ``<AgentLogo>`` root): an SVG in colours of its own,
which a client shows beside the app's name. Each action's logo is an ACTION_LOGO blok
(an ``<ActionLogo>`` root): the same SVG shapes, but painted only with the viewer's palette (``ink``, ``muted``, ``brand``,
``brand-strong``, ``brand-soft``, ``surface``), so the actions of every app look like
one set and follow the viewer's brand and dark mode. A shape that ``animate``s plays
while the logo is hovered, and while the viewer's run of the action is going.

Every logo is drawn on a 24 x 24 grid to read at 16 px. Blok names are the
organization's, so each carries the app's name.
"""

from arkitekt import App

APP = "nd2_converter"

APP_LOGO = f"{APP}_logo"

# An ND2 file, its corner folded, beside a grid of stage positions in three
# fluorescence channels: one multipoint run in, a dataset per point out.
APP_LOGO_SOURCE = """
<AgentLogo viewBox="0 0 64 64" title="nd2-converter">
  <Defs>
    <LinearGradient id="ground" x1="0" y1="0" x2="1" y2="1">
      <Stop offset="0" stopColor="ink" />
      <Stop offset="1" stopColor="ink" />
    </LinearGradient>
  </Defs>
  <Rect x="0" y="0" width="64" height="64" rx="14" fill="url(#ground)" />
  <Path d="M9 11a3 3 0 0 1 3-3h12l7 7v17a3 3 0 0 1-3 3H12a3 3 0 0 1-3-3Z" fill="surface" />
  <Path d="M24 8v5a2 2 0 0 0 2 2h5" fill="muted" />
  <G fill="muted">
    <Rect x="13" y="19" width="12" height="2.5" rx="1.25" />
    <Rect x="13" y="24" width="8" height="2.5" rx="1.25" />
  </G>
  <Path d="M35 12.5h9a5 5 0 0 1 5 5V21" fill="none" stroke="muted" strokeWidth="2.4" strokeLinecap="round" />
  <Path d="M45.5 18.5 49 22l3.5-3.5" fill="none" stroke="muted" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
  <G stroke="ink" strokeWidth="1.6">
    <Rect x="24" y="30" width="13" height="13" rx="2.5" fill="spectrum-green" />
    <Rect x="39" y="30" width="13" height="13" rx="2.5" fill="spectrum-red" />
    <Rect x="24" y="45" width="13" height="13" rx="2.5" fill="spectrum-cyan" />
    <Rect x="39" y="45" width="13" height="13" rx="2.5" fill="spectrum-green" />
  </G>
  <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
</AgentLogo>
"""

ACTION_LOGOS: dict[str, str] = {
    # A file, its corner folded, and the fields of its stage positions lifting out of it.
    "convert_nd2_file_to_datasets": """
<ActionLogo viewBox="0 0 24 24" title="Convert Nikon ND2 file to Datasets">
  <Path d="M3 4a1.5 1.5 0 0 1 1.5-1.5H9l3 3V12a1.5 1.5 0 0 1-1.5 1.5h-6A1.5 1.5 0 0 1 3 12Z" fill="surface" stroke="ink" strokeWidth="1.5" strokeLinejoin="round" />
  <Path d="M9 2.5V5.5H12" fill="none" stroke="ink" strokeWidth="1.5" strokeLinejoin="round" />
  <Path d="M14 4h3a2 2 0 0 1 2 2v1.5" fill="none" stroke="brand-strong" strokeWidth="1.6" strokeLinecap="round" animate="draw" />
  <Path d="M17.25 6 19 7.75 20.75 6" fill="none" stroke="brand-strong" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  <G stroke="ink" strokeWidth="1.4" strokeLinejoin="round">
    <Rect x="10.5" y="11" width="5" height="5" rx="1" fill="brand-strong" animate="float" />
    <Rect x="16.5" y="11" width="5" height="5" rx="1" fill="brand" animate="float" animateDelay="0.15" />
    <Rect x="10.5" y="17" width="5" height="5" rx="1" fill="brand" animate="float" animateDelay="0.3" />
    <Rect x="16.5" y="17" width="5" height="5" rx="1" fill="brand-soft" animate="float" animateDelay="0.45" />
  </G>
</ActionLogo>
""",
}


def logo_of(action: str) -> str:
    """The name of the action logo blok that draws ``action`` (its function's name)."""
    if action not in ACTION_LOGOS:
        raise KeyError(f"No logo is drawn for the action {action!r}")
    return f"{APP}_{action}"


def declare_logos(app: App) -> None:
    """Declares the app's logo, as its default, and every action logo."""
    app.blok(
        APP_LOGO,
        APP_LOGO_SOURCE,
        description="An ND2 file beside a grid of stage positions in three channels.",
        default=True,
    )
    for action, source in ACTION_LOGOS.items():
        app.blok(logo_of(action), source)
