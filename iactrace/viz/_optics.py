import numpy as np
import trimesh

from ..core import euler_to_matrix
from ._meshes import (
    _create_box_mesh,
    _create_cylinder_mesh,
    _create_disk_mesh,
    _create_open_cylinder_mesh,
    _create_oriented_box_mesh,
    _create_polygon_mesh,
    _create_sphere_mesh,
    _create_triangle_mesh,
    _make_double_sided,
    _rigid_transform,
)


def _curved_face_meshes(group):
    """One double-sided curved face mesh per element.

    Shared by mirror groups and refractive lens groups: each element renders
    as a single surface honouring its own aperture and sag.
    """
    meshes = []
    for i in range(len(group)):

        def sag_fn(x, y, _i=i):
            return group.surface.sag_at(_i, x, y)

        mesh = _aperture_face_mesh(
            group.positions[i], group.rotations[i], group.aperture, i, sag_fn=sag_fn
        )
        if mesh is not None:
            meshes.append(_make_double_sided(mesh))
    return meshes


def _aperture_face_mesh(position, rotation_euler, aperture, i, sag_fn=None):
    """Build a single face mesh for element ``i`` of ``aperture``.

    Dispatches on aperture type so callers (mirrors, refractive lenses,
    slab caps) don't need to special-case disks vs. polygons.
    """
    from ..core.apertures import DiskAperture, PolygonAperture

    if isinstance(aperture, DiskAperture):
        return _create_disk_mesh(
            position,
            rotation_euler,
            radius=float(aperture.radii[i]),
            sag_fn=sag_fn,
            inner_radius=float(aperture.inner_radii[i]),
        )
    if isinstance(aperture, PolygonAperture):
        return _create_polygon_mesh(
            position,
            rotation_euler,
            vertices_2d=np.asarray(aperture.vertices[i]),
            sag_fn=sag_fn,
        )
    raise TypeError(f"Unsupported aperture type: {type(aperture).__name__}")


def _aperture_slab_mesh(position, rotation_euler, aperture, i, thickness, sections=32):
    """Extrude element ``i`` of ``aperture`` along the local Z-axis.

    Produces a cylinder for ``DiskAperture`` and a prism for
    ``PolygonAperture``. The front face sits at ``position`` (local
    z = 0) and the back face at local z = +thickness, matching the
    physics convention in :func:`iactrace.core.interactions.refract_slab`
    where ``position`` is the entry point on the front surface.
    """
    from ..core.apertures import DiskAperture, PolygonAperture

    if thickness < 1e-10:
        return None

    if isinstance(aperture, DiskAperture):
        radius = float(aperture.radii[i])
        if radius < 1e-10:
            return None
        mesh = trimesh.creation.cylinder(radius=radius, height=thickness, sections=sections)
        mesh.apply_translation([0.0, 0.0, thickness / 2.0])
    elif isinstance(aperture, PolygonAperture):
        verts_2d = np.asarray(aperture.vertices[i])
        n_verts = len(verts_2d)
        faces_2d = np.array([[0, k, k + 1] for k in range(1, n_verts - 1)])
        mesh = trimesh.creation.extrude_triangulation(verts_2d, faces_2d, height=thickness)
    else:
        raise TypeError(f"Unsupported aperture type: {type(aperture).__name__}")

    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    mesh.apply_transform(_rigid_transform(rot_matrix, np.asarray(position)))
    return mesh


def _get_mirror_meshes(group):
    """Curved face mesh per mirror (each with its own curvature/conic/aspherics)."""
    return _curved_face_meshes(group)


def _get_lens_meshes(group):
    """Get list of lens meshes from group.

    Refractive elements render as a single curved face (one per element);
    slab elements render as a volume extruded along the local Z-axis.
    Both honour the element's aperture, so polygonal lenses and windows
    are supported alongside circular ones.
    """
    from ..core.interactions import RefractInteraction, SlabInteraction

    if isinstance(group.interaction_module, RefractInteraction):
        return _curved_face_meshes(group)

    if isinstance(group.interaction_module, SlabInteraction):
        thickness = np.asarray(group.interaction_module.thickness)
        meshes = []
        for i in range(len(group)):
            mesh = _aperture_slab_mesh(
                group.positions[i], group.rotations[i], group.aperture, i, float(thickness[i])
            )
            if mesh is not None:
                meshes.append(mesh)
        return meshes

    return []


def _get_obstruction_meshes(group):
    """Get list of obstruction meshes from group."""
    from ..core.obstructions import (
        BoxGroup,
        CylinderGroup,
        OpenCylinderGroup,
        OrientedBoxGroup,
        SphereGroup,
        TriangleGroup,
    )

    meshes = []
    if isinstance(group, CylinderGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        r = np.asarray(group.r)
        for i in range(len(group)):
            mesh = _create_cylinder_mesh(p1[i], p2[i], r[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, OpenCylinderGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        r = np.asarray(group.r)
        for i in range(len(group)):
            mesh = _create_open_cylinder_mesh(p1[i], p2[i], r[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, BoxGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        for i in range(len(group)):
            mesh = _create_box_mesh(p1[i], p2[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, SphereGroup):
        centers = np.asarray(group.centers)
        radii = np.asarray(group.radii)
        for i in range(len(group)):
            mesh = _create_sphere_mesh(centers[i], radii[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, OrientedBoxGroup):
        centers = np.asarray(group.centers)
        half_extents = np.asarray(group.half_extents)
        rotations = np.asarray(group.rotations)
        for i in range(len(group)):
            mesh = _create_oriented_box_mesh(centers[i], half_extents[i], rotations[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, TriangleGroup):
        v0 = np.asarray(group.v0)
        v1 = np.asarray(group.v1)
        v2 = np.asarray(group.v2)
        for i in range(len(group)):
            mesh = _create_triangle_mesh(v0[i], v1[i], v2[i])
            if mesh is not None:
                meshes.append(mesh)

    return meshes
