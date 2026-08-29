# ForestFlight Lab design specification

The simulator is a scientific command surface, not a game skin. The Three.js
viewport owns the main visual hierarchy; one narrow telemetry rail presents
only measured planner data.

- Background: `#07110e`; surfaces: `#0c1814`; separators: `#20382d`.
- Safe path/status: `#69e79d`; reference path: `#4f91a6`; newly perceived
  obstacle: `#f0a24a`; unsafe result: `#f26f5b`.
- Typography: native UI sans fallbacks, tabular numerals for timing data.
- Container model: full-bleed canvas plus a flat evidence rail; no nested card
  dashboard.
- Core interaction: density → seed → playback/camera → measured verdict.
- Collision geometry is code-native and deterministic. Trunk cylinders,
  capsule branches, multi-sphere crowns, and the safety envelope use the same
  planner coordinates and dimensions shown in the viewport.
