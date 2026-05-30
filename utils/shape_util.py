import numpy as np
import trimesh
import networkx as nx
import open3d as o3d
from pathlib import Path
import torch
import warnings
import igl
import os

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors


from sklearn.neighbors import NearestNeighbors

from utils.downsample_mesh import decimate_mesh

warnings.simplefilter(action="ignore", category=FutureWarning)


def compute_cosine_similarity_batch(feat_x, feat_y):
    """
    Compute cosine similarity between feature batches using numpy

    Args:
        feat_x: shape [ n_points_x, feature_dim]
        feat_y: shape [ n_points_y, feature_dim]

    Returns:
        similarity: shape [ n_points_x, n_points_y]
    """
    # Normalize features
    feat_x_normed = feat_x / np.linalg.norm(feat_x, axis=-1, keepdims=True)
    feat_y_normed = feat_y / np.linalg.norm(feat_y, axis=-1, keepdims=True)

    assert np.allclose(feat_x_normed, feat_x)
    assert np.allclose(feat_y_normed, feat_y)

    # Batch matrix multiplication
    similarity = feat_x_normed @ feat_y_normed.T

    return similarity


def read_shape(file, as_cloud=False):
    """
    Read mesh from file.

    Args:
        file (str): file name
        as_cloud (bool, optional): read shape as point cloud. Default False
    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3] or None
    """
    if as_cloud:
        verts = np.asarray(o3d.io.read_point_cloud(file).points)
        faces = None
    else:
        mesh = o3d.io.read_triangle_mesh(file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    return verts, faces


def write_off(file, verts, faces):
    with open(file, "w") as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")


def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = min(500, verts.shape[0] // 2)

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), "Graph not connected"
    vertex_adjacency_matrix = nx.adjacency_matrix(
        vertex_adjacency, range(verts.shape[0])
    )
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(
        verts, n_neighbors=NN, mode="distance", include_self=False
    )
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[
        vertex_adjacency_matrix != 0
    ]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print("Inf number in geodesic distance. Increase NN.")
    return geodesic_x


def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err


def all_nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.topk(feat_x.shape[0], dim=dim, largest=False).indices
    return p2p, dist


def edge_comb_to_p2p(
    edge_comb,
    size_VX,
    size_VY,
    adj_matrix_y=None,
    adj_matrix_x=None,
    neighborhood_ring_size=0,
):
    """
    taken from https://github.com/vikiehm/gc-ppsm/blob/main/processing/gen_high_res_mat.py and adjusted to edges
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    # p2pX, p2pY, p2pXFirst, p2pYFirst = p2p_upsample.edge_comb_to_p2p(edge_comb, size_VX, size_VY, adj_matrix_y.todense(), adj_matrix_x.todense(), neighborhood_ring_size)
    p2pX = {}
    p2pY = {}
    p2pXFirst = -1 * np.ones((size_VX, 1))
    p2pYFirst = -1 * np.ones((size_VY, 1))
    for curr_comb in edge_comb:
        for j in range(2):
            comb_x = curr_comb[j]
            comb_y = curr_comb[j + 2]
            add_vertices_x = get_neighboring_vertices(
                adj_matrix_x.todense(), comb_x, neighborhood_ring_size
            )
            add_vertices_y = get_neighboring_vertices(
                adj_matrix_y.todense(), comb_y, neighborhood_ring_size
            )
            for add_vertex_x in add_vertices_x:
                if add_vertex_x not in p2pX:
                    p2pX[add_vertex_x] = add_vertices_y
                else:
                    p2pX[add_vertex_x] = p2pX[add_vertex_x] + add_vertices_y
            p2pXFirst[comb_x] = comb_y
            for add_vertex_y in add_vertices_y:
                if add_vertex_y not in p2pY:
                    p2pY[add_vertex_y] = add_vertices_x
                else:
                    p2pY[add_vertex_y] = p2pY[add_vertex_y] + add_vertices_x
            p2pYFirst[comb_y] = comb_x
    return p2pX, p2pY, p2pXFirst, p2pYFirst


def get_neighboring_vertices(adj_matrix, curr_v, neighborhood_ring_size=0):
    """
    Get neighboring vertices for a given vertex
    Args:
        adj_matrix: adjacency matrix of the graph
        p2p: point-to-point map (shape y -> shape x)
        curr_v: current vertex
        neighborhood_ring_size: size of the neighborhood ring
    Returns:
        neighboring_vertices: set of neighboring vertices
    """
    curr_vs = [curr_v]

    for _ in range(neighborhood_ring_size):
        new_vs = []
        for v in curr_vs:
            neighbors = np.nonzero(adj_matrix[v])[1]
            new_vs.extend(neighbors)
        curr_vs = curr_vs + new_vs
        curr_vs = list(np.unique(curr_vs))
    return curr_vs


def get_high_res_corres(p2pFirstX, p2p, VX_high, highToLowX, highToLowY, p2p_high):
    for i in range(VX_high.shape[0]):
        low_X_idx = highToLowX[i]
        if p2pFirstX[low_X_idx] != -1:
            low_Y_idx = p2p[low_X_idx.item()]
            for curr_low_Y_idx in low_Y_idx:
                # get all idx in highToLowY that have value low_Y_idx
                highToLowY_idx = torch.nonzero(highToLowY == curr_low_Y_idx).flatten()
                highToLowY_idx = highToLowY_idx.unique()
                p2p_high[i, highToLowY_idx.long()] = 1
    return p2p_high


def compute_low_res_corr_with_averaging(
    vx2VX, vy2VY, DX, DY, low_res_features_via_averaging_neighbourhood
):
    num_nearest_neighbours = low_res_features_via_averaging_neighbourhood
    nn_x_heigh_res = np.argsort(DX[vx2VX], axis=1)[:, :num_nearest_neighbours]
    nn_y_heigh_res = np.argsort(DY[vy2VY], axis=1)[:, :num_nearest_neighbours]
    return nn_x_heigh_res, nn_y_heigh_res


def compute_edge_costs(feat_x, feat_y, vx, vx2VX, vy, vy2VY, DX=None, DY=None):

    down_feat_x = feat_x[vx2VX]
    down_feat_y = feat_y[vy2VY]
    edge_costs = compute_cosine_similarity_batch(down_feat_x, down_feat_y)
    edge_costs = -edge_costs  # convert to distance
    edge_costs = edge_costs + 1  # convert to [0,
    edge_costs = edge_costs / 2  # normalize to [0, 1]
    closest_indices_lowres = np.argmin(edge_costs, axis=0)
    return edge_costs, closest_indices_lowres


def high_to_low_res(dist_x, dist_y, vx2VX, vy2VY, VX, VY):
    VX2vx = np.ones(VX.shape[0], dtype=np.int32) * -1
    VY2vy = np.ones(VY.shape[0], dtype=np.int32) * -1
    # where vx2VX == index of VX, set VX2vx to index
    for i in range(vx2VX.shape[0]):
        VX2vx[vx2VX[i]] = i
    for i in range(vy2VY.shape[0]):
        VY2vy[vy2VY[i]] = i

    not_set_x = VX2vx == -1
    not_set_y = VY2vy == -1

    for i in range(VX2vx.shape[0]):
        if VX2vx[i] == -1:
            dist_new = dist_x[i, :].copy()
            dist_new[not_set_x] = np.inf
            min_idx = np.argmin(dist_new)
            VX2vx[i] = VX2vx[min_idx]
    for i in range(VY2vy.shape[0]):
        if VY2vy[i] == -1:
            dist_new = dist_y[i, :].copy()
            dist_new[not_set_y] = np.inf
            min_idx = np.argmin(dist_new)
            VY2vy[i] = VY2vy[min_idx]
    return VX2vx, VY2vy


def compute_geodesic_dist_matrices(
    dist_matrix_folder, shape_1, shape_2, VX, FX, VY, FY
):
    y_dist_path = dist_matrix_folder / f"{shape_2}.npz"
    x_dist_path = dist_matrix_folder / f"{shape_1}.npz"
    if not dist_matrix_folder.exists():
        dist_matrix_folder.mkdir(parents=True, exist_ok=True)
    if not x_dist_path.exists():
        dist_x = compute_geodesic_distmat(VX, FX)
        np.savez(x_dist_path, dist=dist_x)
    else:
        dist_x = np.load(x_dist_path, allow_pickle=True)["dist"]

    if not y_dist_path.exists():
        dist_y = compute_geodesic_distmat(VY, FY)
        np.savez(y_dist_path, dist=dist_y)
    else:
        dist_y = np.load(y_dist_path, allow_pickle=True)["dist"]
    return dist_x, dist_y


def shape_loader(filename1, filename2, shape_loader_opts):
    vert_np_x, face_np_x = read_shape(filename1)
    vert_np_y, face_np_y = read_shape(filename2)

    VX_orig = vert_np_x
    FX_orig = face_np_x
    VY_orig = vert_np_y
    FY_orig = face_np_y

    nfacesX = min(shape_loader_opts["num_faces"], len(FX_orig))
    nfacesY = min(shape_loader_opts["num_faces"], len(FY_orig))

    ratio = FY_orig.shape[0] / FX_orig.shape[0]
    nfacesX = round(shape_loader_opts["num_faces"] / (ratio + 1))
    nfacesX = min(nfacesX, len(FX_orig))
    nfacesY = round(nfacesX * ratio)
    nfacesY = min(nfacesY, len(FY_orig))

    VX, FX = decimate_mesh(VX_orig.copy(), FX_orig.copy(), nfacesX)
    VY, FY = decimate_mesh(VY_orig.copy(), FY_orig.copy(), nfacesY)

    mean_area_faces_X = igl.doublearea(VX, FX).mean()
    mean_area_faces_Y = igl.doublearea(VY, FY).mean()

    if mean_area_faces_Y > mean_area_faces_X:
        area_ratio = mean_area_faces_Y / mean_area_faces_X
        nfacesX = round(shape_loader_opts["num_faces"] / (area_ratio + 1))
        nfacesX = min(nfacesX, len(FX_orig))
        nfacesY = round(nfacesX * area_ratio)
        nfacesY = int(min(nfacesY, len(FY_orig)))
        VX, FX = decimate_mesh(VX_orig.copy(), FX_orig.copy(), nfacesX)
        VY, FY = decimate_mesh(VY_orig.copy(), FY_orig.copy(), nfacesY)
        mean_area_faces_X = igl.doublearea(VX, FX).mean()
        mean_area_faces_Y = igl.doublearea(VY, FY).mean()

    print(f"Final number of faces in X: {FX.shape[0]}, Y: {FY.shape[0]}")
    print(f"Final mean area of faces in X: {mean_area_faces_X}, Y: {mean_area_faces_Y}")

    dist_matrix_folder = Path(shape_loader_opts["geodist_path"])
    shape_1 = os.path.basename(filename1).split(".")[0]
    shape_2 = os.path.basename(filename2).split(".")[0]

    dist_x, dist_y = compute_geodesic_dist_matrices(
        dist_matrix_folder, shape_1, shape_2, VX_orig, FX_orig, VY_orig, FY_orig
    )

    idx_vx_in_orig = knn_search(VX, VX_orig)
    idx_vy_in_orig = knn_search(VY, VY_orig)
    idx_VX_in_vx, idx_VY_in_vy = high_to_low_res(
        dist_x, dist_y, idx_vx_in_orig, idx_vy_in_orig, VX_orig, VY_orig
    )

    return (
        VX_orig,
        FX_orig,
        VX,
        FX,
        idx_vx_in_orig,
        idx_VX_in_vx,
        VY_orig,
        FY_orig,
        VY,
        FY,
        idx_vy_in_orig,
        idx_VY_in_vy,
    )


def knn_search(x, X, k=1):
    """
    find indices of k-nearest neighbors of x in X
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X)
    _, indices = nbrs.kneighbors(x)
    if k == 1:
        return indices.flatten()
    else:
        return indices
