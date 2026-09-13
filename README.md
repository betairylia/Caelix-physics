# Caelix Physics

Caelix Physics is the Caelix physics package and a maintained fork of Unity Physics.

Caelix-owned integration code is in `Caelix.Physics`. The upstream-derived low-level code remains in `Unity.Physics`.

This is an early research and development release.

## Caelix tick integration

`VoxelPhysicsWorld.PrepareSpatialQueries` prepares body membership, mappings, poses
and voxel bounds after the topology boundary, then builds the shared physics BVH.
`CollectAutomataCandidates` returns entity indices for each required simulation
brick's one-voxel neighbourhood, without enumerating target bricks. Entity/body
membership, staticness and transforms must stay fixed until physics.

`SimulateStep` refreshes collider storage after snapshot commits and prepares motion
from the updated physics properties and forces. It reuses the prepared static tree
when bounds are unchanged. Standalone callers may omit spatial preparation.
`BuildBrickOverlapGraph` remains a separate post-physics query and ensures the
collision world has the resulting poses before collecting overlaps. Candidate lists
from before automata must not be reused after movement or geometry changes.

## Related Repositories

- [Caelix-Core](https://github.com/betairylia/Caelix-Core) — Basic voxel data layouts + I/O.
- [Caelix-physics](https://github.com/betairylia/Caelix-physics) — Voxel fork of Unity Physics.
- [Caelix](https://github.com/betairylia/Caelix) — Custom voxel rendering & physics engine backend.

---

## About Unity Physics
The Unity Physics package provides a stateless, high-performance constrained rigid body simulation.

## Installing Unity Physics
To install this package, follow the instructions in the [Getting Started documentation](Documentation~/getting-started-installation.md).

## Using Unity Physics
For information on how to use the Unity Physics package, see the [User manual](Documentation~/index.md).
