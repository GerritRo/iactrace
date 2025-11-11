"""Telescope configuration and compilation."""

import jax
import jax.numpy as jnp
from jax import vmap
from .obstructions import Cylinder, Box
from .simulation import CompiledSimulation
from ..utils.transforms import look_at_rotation
from ..core.surfaces import spherical_surface, parabolic_surface


class Telescope:
    """
    IACT telescope configuration.

    This class provides an OOP interface for building telescope configurations.
    Call compile() to extract all data into pure JAX arrays for efficient simulation.
    """

    def __init__(self, mirror_positions, focal_length=15.0, dish_radius=15.0,
                 surface_type='spherical', alignment_perturbation=0.001):
        """
        Args:
            mirror_positions: Mirror facet positions (N, 2) or (N, 3) with diameter
                             Format: [x, y] or [x, y, diameter]
            focal_length: Focal length for parabolic surfaces
            dish_radius: Radius of curvature for spherical surfaces (Davies-Cotton)
            surface_type: 'spherical' or 'parabolic'
            alignment_perturbation: Random alignment error scale (default 0.001)
        """
        self.mirror_positions = jnp.array(mirror_positions)
        self.focal_length = focal_length
        self.dish_radius = dish_radius
        self.surface_type = surface_type
        self.alignment_perturbation = alignment_perturbation
        self.obstructions = []

    def add_obstruction(self, obstruction):
        """
        Add an obstruction (Cylinder or Box) to the telescope.

        Args:
            obstruction: Cylinder or Box object
        """
        self.obstructions.append(obstruction)

    def _compute_transforms(self, key=None):
        """
        Compute mirror transforms for Davies-Cotton geometry.

        Returns:
            translations: Mirror positions in 3D (n_mirrors, 3)
            rotations: Mirror rotation matrices (n_mirrors, 3, 3)
        """
        # Compute mirror positions on dish
        if self.surface_type == 'spherical':
            Tmirrors, _ = spherical_surface(self.dish_radius, self.mirror_positions[:, :2])
            align_z = 2*self.dish_radius
        elif self.surface_type == 'parabolic':
            Tmirrors, _ = parabolic_surface(self.focal_length, self.mirror_positions[:, :2])
            align_z = 2*self.focal_length
        else:
            raise ValueError(f"Unknown surface type: {self.surface_type}")

        # Compute rotations to look at focal point (with optional perturbation)
        if key is not None and self.alignment_perturbation > 0:
            align_target = jax.random.normal(key, Tmirrors.shape) * self.alignment_perturbation + \
                          jnp.array([0, 0, align_z])
        else:
            align_target = jnp.tile(jnp.array([0, 0, align_z]), (len(Tmirrors), 1))

        Rmirrors = vmap(look_at_rotation, in_axes=(0, 0))(Tmirrors, align_target)

        return Tmirrors, Rmirrors

    def _extract_obstructions(self):
        """
        Extract obstruction data into arrays.

        Returns:
            Tuple of (cyl_p1, cyl_p2, cyl_radius, box_centers, box_sizes)
        """
        cylinders = [o for o in self.obstructions if isinstance(o, Cylinder)]
        boxes = [o for o in self.obstructions if isinstance(o, Box)]

        if cylinders:
            cyl_p1 = jnp.array([c.p1 for c in cylinders])
            cyl_p2 = jnp.array([c.p2 for c in cylinders])
            cyl_radius = jnp.array([c.radius for c in cylinders])
        else:
            cyl_p1 = jnp.zeros((0, 3))
            cyl_p2 = jnp.zeros((0, 3))
            cyl_radius = jnp.zeros((0,))

        if boxes:
            box_p1 = jnp.array([b.p1 for b in boxes])
            box_p2 = jnp.array([b.p2 for b in boxes])
        else:
            box_p1 = jnp.zeros((0, 3))
            box_p2 = jnp.zeros((0, 3))

        return cyl_p1, cyl_p2, cyl_radius, box_p1, box_p2

    def compile(self, integrator, sensor, source_type='point',
                sensor_plane=(jnp.array([0., 0., 15.]), jnp.array([0., 0., -1.])),
                alignment_key=None, sampling_key=None):
        """
        Compile telescope into a pure JAX simulation function.

        This extracts all OOP structure into arrays and returns a JIT-compiled
        function that can be composed with other JAX code.

        Args:
            integrator: Integrator object (e.g., MCIntegrator)
            sensor: Sensor object (e.g., SquareSensor, HexagonalSensor)
            source_type: 'point' or 'infinity'
            sensor_plane: Tuple of (plane_position, plane_normal)
            alignment_key: JAX random key for mirror alignment perturbation
            sampling_key: JAX random key for mirror sampling

        Returns:
            CompiledSimulation object (pure JAX pytree + function)
        """
        if sampling_key is None:
            sampling_key = jax.random.key(0)

        # Sample mirrors
        mirror_array = integrator.sample_mirrors(self, sampling_key)

        # Compute transforms
        Tmirrors, Rmirrors = self._compute_transforms(alignment_key)

        # Extract obstructions
        cyl_p1, cyl_p2, cyl_radius, box_p1, box_p2 = self._extract_obstructions()
        
        # Create compiled simulation
        return CompiledSimulation(
            mirror_points=mirror_array.points,
            mirror_normals=mirror_array.normals,
            mirror_positions=Tmirrors,
            mirror_rotations=Rmirrors,
            cyl_p1=cyl_p1,
            cyl_p2=cyl_p2,
            cyl_radius=cyl_radius,
            box_p1=box_p1,
            box_p2=box_p2,
            sensor_plane_pos=sensor_plane[0],
            sensor_plane_normal=sensor_plane[1],
            sensor_config=sensor.to_config(),
            accumulate_fn=sensor.make_accumulate_fn()
        )

    @classmethod
    def from_csv(cls, mirror_file, mast_file=None, **kwargs):
        """
        Load telescope from CSV files (HESS format).

        Args:
            mirror_file: Path to mirror positions CSV
            mast_file: Path to mast/cylinder obstructions file (optional)
            **kwargs: Additional arguments passed to Telescope constructor

        Returns:
            Telescope object
        """
        import pandas as pd

        # Load mirrors
        df = pd.read_csv(mirror_file, comment='#', sep=r'\s+')
        if any(col in df.columns for col in [',', '']):
            # Clean comma artifacts
            df = df.apply(lambda x: x.astype(str).str.replace(',', ''))
            df = df.astype(float)

        # Assume format is x, y, diameter (convert cm to m)
        mirror_positions = jnp.array(df[['y', 'x', 'p']].values) / 100

        telescope = cls(mirror_positions, **kwargs)

        # Load masts/cylinders if provided
        if mast_file is not None:
            df = pd.read_csv(mast_file, comment='#', sep=r'\s+')
            if any(col in df.columns for col in [',', '']):
                df = df.apply(lambda x: x.astype(str).str.replace(',', ''))
                df = df.astype(float)

            cyl_points1 = jnp.array(df[['x1', 'y1', 'z1']].values) * 0.01
            cyl_points2 = jnp.array(df[['x2', 'y2', 'z2']].values) * 0.01
            cyl_radius = jnp.array(df[['d']].values / 2) * 0.01

            for i in range(len(cyl_points1)):
                telescope.add_obstruction(Cylinder(
                    p1=cyl_points1[i],
                    p2=cyl_points2[i],
                    radius=float(cyl_radius[i])
                ))

        return telescope
