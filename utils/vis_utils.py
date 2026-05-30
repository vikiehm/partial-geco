import numpy as np

def rotate_shape(verts):
    rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    verts = verts @ rotation_matrix.T

    # and then rotate both shapes by -90 degrees around the x-axis
    rotation_matrix_y = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    verts = verts @ rotation_matrix_y.T
    return verts


def vis_p2p(
    verts_source,
    verts_target,
    faces_source,
    faces_target,
    corres,
    corres_source,
    dataset,
    save_path="correspondences.png",
):
    """
    Visualize point-to-point correspondences between two meshes.

    Args:
        verts_source (np.ndarray): Vertices of the source mesh
        verts_target (np.ndarray): Vertices of the target mesh
        faces_source (np.ndarray): Faces of the source mesh
        faces_target (np.ndarray): Faces of the target mesh
        corres (np.ndarray): Correspondence indices between source and target vertices
    """
    import polyscope as ps

    ps.init()

    y_shift = np.array(
        [0, 0.7, 0]
    )  # Offset shape Y to the right for better visualization
    verts_source = verts_source + y_shift
    # make copies of the vertices and faces to avoid modifying the original data
    verts_source = verts_source.copy()
    verts_target = verts_target.copy()
    faces_source = faces_source.copy()
    faces_target = faces_target.copy()

    verts_source = rotate_shape(verts_source)
    verts_target = rotate_shape(verts_target)

    poly_shape_source = ps.register_surface_mesh("source", verts_source, faces_source)
    poly_shape_target = ps.register_surface_mesh("target", verts_target, faces_target)
    color_source = verts_source
    color_source = color_source - np.min(color_source)
    color_source = color_source / np.max(color_source)
    color_source[corres_source == -1] = np.ones(3) * 0.7
    color_target = color_source[corres]
    color_target[corres == -1] = np.ones(3) * 0.7
    poly_shape_source.add_color_quantity(
        "correspondences", color_source, defined_on="vertices", enabled=True
    )
    poly_shape_target.add_color_quantity(
        "correspondences", color_target, defined_on="vertices", enabled=True
    )

    ps.screenshot(save_path)
    ps.remove_all_structures()
