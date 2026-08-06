# Changelog

## v0.10.0 (2026-08-06)

### Feat

- **trajectory**: unified treatment of trajectories between components, fixed bugs in viz
- **viz,-core**: added the ability to save trajectories for the optical trace, added methods to visualize traces for optics and detection_chain

### Fix

- **docs,-core**: fixed documentation and removed bugs from formatting. fixed some problems with normalization and polygonal apertures
- **core**: fixed issue with surface relocalization
- **core,-viz**: fixed a bug in the intersection logic, leading to the wrong root being chosen for conics. also made float32 computation more stable by propagating to local object reference. improved visualization of traced rays. refactored telescope3d by breaking up into files

### Refactor

- **ruff**: reformat and fixes with ruff
- **ruff,-tests**: reformated with ruff, changed default setting of laye_vs_eager test to not be on pixel boundary

## v0.9.1 (2026-08-06)

### Fix

- **core, docs**: fixed angle conventions, more info

## v0.9.0 (2026-07-18)

### BREAKING CHANGE

- Closes #11

### Feat

- **core,tests,camera**: refactored tests to be more concentrated and better motivated, refactored to reduce code duplication and increase polymorphism, restructured camera module
- **camera**: created a combined tracing chain, including photosensor surface, allowing for peeking
- **core**: added an alive flag to ray bundles, to avoid confusing properties regarding value and hit mask
- **camera**: added okumura cone prototype
- **camera,configs,io,viz,tests**: implemented a detection chain that allows for concentrators and complicated photosensors
- **core**: added angle dependent coatings via tabulation

### Fix

- **configs**: changed the ordering of the modules to reflective of the way it is done in sim_telarray
- **core,-surfaces**: added more surface types, fixed shadowing bug for tracing
- **configs**: improved telescope shadowing (no baffling instead of monolithic camera enclosure, to avoid incorrect self shadowing. also added a best guess okumura cone to LSTCam
- **configs**: Added winston cones to flashcam, nectarcam and HESS2cam
- **configs,viz**: added winston cone to HESS1U, fixed relative position of mirrors in the context of coordinate conventions, added visualization of the sensor grid
- **core**: fix that there is no shadowing applied in the last leg to the focal plane, improved focal plane analysis
- **camera**: fixed roundtripping issues with camera and coatings
- **camera,core,io**: fixed bugs regarding mirroring, io and camera rays
- **io**: fixed bsdf not bein correctly serialized

### Refactor

- **refactor**: refactored camera module, slimmed tests, better io for surfaces
- **format,-doc**: ruff formatting and small doc changes regarding conventions

## v0.8.0 (2026-06-11)

### Feat

- **opl**: make opl behave as expected, weighted with index of refraction, add distance to source or initial mirror

### Fix

- **viz**: fixed 3D visualization to correctly handle transparency in jupyter notebooks

## v0.7.0 (2026-05-12)

### Feat

- **configs**: added configs for the SST telescope with CHECS camera
- **io,viz**: enabled both apertures types for slabs and lenses for plotting and io
- **analysis,camera,viz**: improved plotting and combined plotting path for sensors, added analysis tools such as focal plane and fixed multi-sensor groups

### Fix

- **benchmarks**: fixed benchmark to mvoe from MCIntegrator to direct sampling
- **surfaces**: bug in aspheric surface formula squared unnecessarily

### Refactor

- **docs,configs**: adjusted docs to reflect new structure, split configs into telescope/camera, ran ruff
- **iactrace**: major refactor of architecture, better support for lens optics
- **utils**: removed utils path, unneccessary for new sampling method

## v0.6.1 (2026-05-06)

### Fix

- **viz**: changed hexshow colorscaling to follow the same logic as squareshow via plt.Normalize
- **operations**: changed mirror displacement to go along the local mirror z axis, similar to simtel_array
- **core**: fixed bug in conic intersection

### Perf

- **core**: bypass NR for pure conics

## v0.6.0 (2026-02-09)

### BREAKING CHANGE

- initial project setup

### Feat

- add changelog
- initial release
