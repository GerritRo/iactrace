import jax
import jax.numpy as jnp
import equinox as eqx


def resample(telescope, integrator, key):
    """Return telescope with resampled mirror groups."""
    keys = jax.random.split(key, len(telescope.mirror_groups))
    new_groups = [integrator.sample_group(g, k) 
                  for g, k in zip(telescope.mirror_groups, keys)]
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def set_sensor(telescope, sensor, idx=0):
    """Return telescope with replaced sensor."""
    new_sensors = list(telescope.sensors)
    new_sensors[idx] = sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def add_obstruction(telescope, obstruction):
    """Return telescope with additional obstruction group."""
    new_groups = telescope.obstruction_groups + [obstruction]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def apply_roughness(telescope, roughness, key):
    """Apply surface roughness to mirror normals."""
    from ..core import perturb_normals
    
    sigma_rad = roughness * jnp.pi / (180.0 * 3600.0)
    keys = jax.random.split(key, len(telescope.mirror_groups))
    new_groups = []
    for group, k in zip(telescope.mirror_groups, keys):
        new_normals = perturb_normals(group.normals, sigma_rad, k)
        new_groups.append(eqx.tree_at(lambda g: g.normals, group, new_normals))
    
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)