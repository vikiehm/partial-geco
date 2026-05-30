import numpy as np
import igl


def get_boundaries(fx, fy, product_space):
    boundary_edges = igl.boundary_facets(fx).astype(np.int32)
    boundary_vertices = np.unique(boundary_edges)
    # combine the same boundary vertices (n x 1) to one edge, such that degenerate_boundary_edges has the size (n x 2)
    degenerate_boundary_edges = np.column_stack((boundary_vertices, boundary_vertices))

    boundary_edges_inverse = boundary_edges[:, [1, 0]]
    mask_boundary = (
        (product_space[:, 2:4][:, None] == boundary_edges[None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_inverse = (
        (product_space[:, 2:4][:, None] == boundary_edges_inverse[None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_all_x = np.logical_or(mask_boundary, mask_boundary_inverse)
    # add also degenerate boundary edges
    mask_boundary_degenerate_x = (
        (product_space[:, 2:4][:, None] == degenerate_boundary_edges[None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_all_x = np.logical_or(mask_boundary_all_x, mask_boundary_degenerate_x)

    boundary_edges_y = igl.boundary_facets(fy).astype(np.int32)
    boundary_vertices_y = np.unique(boundary_edges_y)
    degenerate_boundary_edges_y = np.column_stack(
        (boundary_vertices_y, boundary_vertices_y)
    )
    mask_boundary_y = (
        (product_space[:, 0:2][:, None] == boundary_edges_y[None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_inverse_y = (
        (product_space[:, 0:2][:, None] == boundary_edges_y[:, [1, 0]][None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_degenerate_y = (
        (product_space[:, 0:2][:, None] == degenerate_boundary_edges_y[None, :])
        .all(axis=2)
        .any(axis=1)
    )
    mask_boundary_all_y = np.logical_or(mask_boundary_y, mask_boundary_inverse_y)
    mask_boundary_all_y = np.logical_or(mask_boundary_all_y, mask_boundary_degenerate_y)
    mask_boundary_all = np.logical_or(mask_boundary_all_x, mask_boundary_all_y)

    return mask_boundary_all


def get_overlap_constraint_mask(product_space, S, fy, overlap21_lowres):
    assert S.shape[0] == fy.shape[0] * 3, (
        "Constraint matrix S should have 3 times the number of faces in Y."
    )

    S_edges = np.zeros((S.shape[0], 2), dtype=int)
    overlap_S_mask = np.zeros((S.shape[0],), dtype=float)
    for i in range(S.shape[0]):
        curr_product_space = product_space[S[i].toarray().squeeze() == 1, :]
        if curr_product_space.sum() == 0:
            curr_edge = np.array([-1, -1])
            S_edges[i, :] = curr_edge
        else:
            curr_edge = curr_product_space[:, 0:2]
            S_edges[i, :] = curr_edge[
                0, 0:2
            ]  # take the first edge of the product space

    for i in range(S_edges.shape[0]):
        curr_edge = S_edges[i, :]
        if curr_edge[0] == -1 or curr_edge[1] == -1:
            overlap_S_mask[i] = 0
            continue
        # find corresponding face in Y
        vert0 = curr_edge[0]
        vert1 = curr_edge[1]
        overlap_S_mask[i] = np.mean(overlap21_lowres[[vert0, vert1]])

    return overlap_S_mask


def get_pruned_search_space(vx, vy, p2p_X, p2p_Y, product_space):
    pruned_product_space = np.zeros(product_space.shape[0], dtype=bool)
    for i in range(vx.shape[0]):
        possible_matchings = p2p_X.get(i, [])
        if len(possible_matchings) > 0:
            product_space_i = product_space[product_space[:, 2] == i, 0]
            in_mask = np.isin(product_space_i, possible_matchings)
            pruned_product_space[product_space[:, 2] == i] = (
                pruned_product_space[product_space[:, 2] == i] | in_mask
            )

            product_space_i = product_space[product_space[:, 3] == i, 1]
            in_mask = np.isin(product_space_i, possible_matchings)
            pruned_product_space[product_space[:, 3] == i] = (
                pruned_product_space[product_space[:, 3] == i] | in_mask
            )

    for i in range(vy.shape[0]):
        possible_matchings = p2p_Y.get(i, [])
        if len(possible_matchings) > 0:
            product_space_i = product_space[product_space[:, 0] == i, 2]
            in_mask = np.isin(product_space_i, possible_matchings)
            pruned_product_space[product_space[:, 0] == i] = (
                pruned_product_space[product_space[:, 0] == i] | in_mask
            )

            product_space_i = product_space[product_space[:, 1] == i, 3]
            in_mask = np.isin(product_space_i, possible_matchings)
            pruned_product_space[product_space[:, 1] == i] = (
                pruned_product_space[product_space[:, 1] == i] | in_mask
            )

    return pruned_product_space


def result_to_p2p(result_vec, product_space, verts_low_target, verts_low_source):
    """
    Upsample results from a lower resolution to a higher resolution mesh.

    Args:
        result_vec (np.ndarray): Result vector to upsample
        product_space (object): Product space containing the upsampling method
        verts_low_target (np.ndarray): Vertices of the low resolution target mesh
        faces_low_target (np.ndarray): Faces of the low resolution target mesh

    Returns:
        np.ndarray: Upsampled result vector
    """
    used_product_space = product_space[result_vec > 0.5][:, 0:4]
    p2p_source = np.ones(verts_low_source.shape[0]) * -1
    p2p_target = np.ones(verts_low_target.shape[0]) * -1
    p2p_source[used_product_space[:, 0]] = used_product_space[:, 2]
    p2p_source[used_product_space[:, 1]] = used_product_space[:, 3]
    p2p_target[used_product_space[:, 2]] = used_product_space[:, 0]
    p2p_target[used_product_space[:, 3]] = used_product_space[:, 1]
    return p2p_source.astype(int), p2p_target.astype(int)


def get_high_correspondences(p2pFirstX, VX_high, highToLowX, highToLowY):
    p2p_high = np.ones(VX_high.shape[0]) * -1
    for i in range(VX_high.shape[0]):
        low_X_idx = highToLowX[i]
        if p2pFirstX[low_X_idx] != -1:
            low_Y_idx = [p2pFirstX[low_X_idx.item()]]
            for curr_low_Y_idx in low_Y_idx:
                # get all idx in highToLowY that have value low_Y_idx
                highToLowY_idx = np.unique(np.where(highToLowY == curr_low_Y_idx)[0])
                p2p_high[i] = highToLowY_idx[0] if len(highToLowY_idx) > 0 else -1
    return p2p_high.astype(int)
