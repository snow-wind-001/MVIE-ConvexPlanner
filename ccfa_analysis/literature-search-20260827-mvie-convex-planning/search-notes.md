# Search Notes

## Safe Queries Used

- `Fast Iterative Region Inflation FIRI convex region obstacle-free space trajectory planning`
- `convex safe corridor generation and maximum volume inscribed ellipsoid for collision-free robot trajectory planning`
- `dynamically feasible B-spline trajectory optimization with convex corridor acceleration and jerk constraints for quadrotors`
- `iterative obstacle avoidance path projection waypoint pushing and convex region inflation in 3D motion planning`
- `convex safe flight corridor trajectory optimization GCOPTER FASTER EGO-Planner quadrotor`
- `maximum volume inscribed ellipsoid safe corridor trajectory optimization robot motion planning`
- `B-spline control point acceleration jerk constraints convex corridor trajectory optimization quadrotor`
- `Deits Tedrake Computing Large Convex Regions of Obstacle-Free Space Through Semidefinite Programming WAFR 2015 official`

No repository sentence, source file, author identity, or unpublished result was submitted as a search query.

## Sources Checked

- Firecrawl Research semantic search over paper abstracts.
- Related-paper expansion in similar and reference modes from FIRI and B-spline trajectory anchors.
- In-body verification for FIRI, GCOPTER, Fast-Planner, direct point-cloud polytope generation, and coupled convex-cover optimization.
- Stable arXiv records, the FIRI DOI, and the Springer proceedings record for IRIS.

## Excluded Sources

- Policy-excluded and low-traceability sources were omitted from the final paper table.
- Discovery-only mirrors and search snippets were not used for load-bearing novelty claims.

## Unknowns

- Venue/publisher status was not independently reverified for every arXiv entry; those rows are labeled as public preprints.
- The search did not establish that analytic mixed sphere/cylinder/cuboid support functions are novel; that rescue direction requires a focused second search.
- No public benchmark protocol exactly matching the repository's claimed 30-scene evaluation was found inside the repository.

## Handoff Notes

- For idea review: direct novelty is low because region inflation, corridor optimization, B-spline dynamics, and coupled cover/trajectory design all have close prior art.
- For experiment design: reproduce FIRI, GCOPTER, and one B-spline baseline under shared maps, seeds, dynamic limits, and timing hardware.
- For writing: position any future paper around one verified new mechanism; do not frame the module stack itself as the contribution.
