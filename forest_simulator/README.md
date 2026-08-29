# ForestFlight Lab

Three.js interactive evidence viewer for the repository's forest-flight
avoidance test. The browser does not reimplement the planner: it renders JSON
produced by `generate_forest_scenarios.py` and exposes safe, failed and timed
out outcomes without changing them.

## Run

```bash
npm install
npm run generate:data
npm run dev
```

Open `http://127.0.0.1:5173/`. Use the density tabs and seed selector to inspect
all scenarios. The checked-in optimized dataset contains 30/30 verified-safe
realtime repairs; the interface still preserves safe-hold rendering for future
scenarios where a verified repair is unavailable inside the 20 ms budget.

## Protocol

1. Generate a reproducible 12/20/28-tree forest. Every tree contains an
   analytic trunk cylinder, three finite 3-D branch capsules, and at least
   three colliding canopy spheres.
2. Run full FIRI/MVIE/SLSQP to produce and verify the low-frequency reference.
3. Resample the reference at no more than 4 m spacing, matching a local control
   reference rather than a few very long global segments.
4. Add one newly perceived route-crossing branch or full tree.
5. Run one bounded spherical realtime repair and independently verify the
   returned polyline against all trunks, slanted branches and canopy spheres
   plus the 0.30 m margin.

The Three.js geometry matches the Python collision records: branch cylinders
include spherical capsule ends, and every rendered canopy lobe is a colliding
sphere rather than visual-only context.

## Build

```bash
npm run build
```
