# Changelog

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
