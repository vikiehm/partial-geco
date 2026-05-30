# from smm_dijkstra import ShapeMatchModelDijkstra
import gurobipy as gp
from gurobipy import GRB
import time
import scipy.sparse as sp
from geco import product_graph_generator, get_surface_cycles
from utils.shape_util import shape_loader
import polyscope as ps
import igl
import torch
from utils.vis_utils import (
    vis_p2p,
)
import numpy as np
import os
from pathlib import Path
from utils.p2p_geco_utils import (
    get_boundaries,
    get_overlap_constraint_mask,
    get_pruned_search_space,
    get_high_correspondences,
    result_to_p2p,
)
from utils.shape_util import (
    compute_geodesic_distmat,
    edge_comb_to_p2p,
    compute_edge_costs,
    high_to_low_res,
)

headless = True  # Set to False if you want to see the polyscope window
ps.set_allow_headless_backends(headless)
ps.init()


def coarse_to_fine(
    old_nr, new_nr, map_filename, lamda=0.3, neighborhood_ring_size=2, dataset="cp2p"
):
    time_limit = 1800  # 30 minutes time limit for each optimization
    max_depth = 2  # multiples of 2, quadratic size increase
    resolve_coupling = False
    base_dir = Path(__file__).resolve().parents[1]
    results_filename = "result.npz"
    factor_slack_S = lamda
    factor_slack_vx = lamda

    load_folder = base_dir / "results" / f"cp2p_{old_nr}"
    input_result_file = load_folder / map_filename / results_filename
    if not input_result_file.exists():
        raise FileNotFoundError(
            f"No input results found in {load_folder}. Expected {map_filename}/{results_filename}."
        )
    off_folder = str(base_dir / "example_data" / "shapes")
    results_folders = str(base_dir / "results" / f"cp2p_{new_nr}")
    feat_folder = str(base_dir / "example_data" / "features") + "/"
    dist_path = base_dir / "geo_matrices"
    shape_1 = map_filename.split("_")[0]
    shape_2 = map_filename.split("_")[1]
    filename1 = os.path.join(off_folder, shape_1 + ".off")
    filename2 = os.path.join(off_folder, shape_2 + ".off")

    results = np.load(load_folder / map_filename / results_filename)

    vx_org = results.get("vx")
    vy_org = results.get("vy")
    fx_org = results.get("fx")
    fy_org = results.get("fy")
    edge_matching = results.get("matching_edges")

    adj_matrix_x = igl.adjacency_matrix(fx_org)
    adj_matrix_y = igl.adjacency_matrix(fy_org)

    print("Starting to compute p2p correspondences...")

    p2pY_org, p2pX_org, _, _ = edge_comb_to_p2p(
        edge_matching,
        vy_org.shape[0],
        vx_org.shape[0],
        adj_matrix_x=adj_matrix_y,
        adj_matrix_y=adj_matrix_x,
        neighborhood_ring_size=neighborhood_ring_size,
    )

    print("Computing p2p correspondences done.")

    # generate results folder if it does not exist
    curr_results_folder = os.path.join(results_folders, map_filename)
    if not os.path.exists(curr_results_folder):
        os.makedirs(curr_results_folder)

    print("Using files:", filename1, filename2)
    shape_opts = {
        "num_faces": new_nr,
        "partial_cp2p": True,
        "geodist_path": dist_path,
    }

    ## Load and downsample shapes and compute spidercurve on shape X
    VX, FX, vx, fx, vx2VX, VX2vx, VY, FY, vy, fy, vy2VY, VY2vy = shape_loader(
        filename1, filename2, shape_opts
    )

    closest_vertices = np.linalg.norm(vy_org - vy[:, None], axis=2)
    # # # get the indices of the closest features
    vy_org2vy = np.argmin(closest_vertices, axis=0)
    closest_vertices = np.linalg.norm(vx_org - vx[:, None], axis=2)

    vx_org2vx = np.argmin(closest_vertices, axis=0)
    dist_x = compute_geodesic_distmat(vx, fx)
    dist_y = compute_geodesic_distmat(vy, fy)
    vx2vx_org, vy2vy_org = high_to_low_res(
        dist_x=dist_x, dist_y=dist_y, vx2VX=vx_org2vx, vy2VY=vy_org2vy, VX=vx, VY=vy
    )

    p2p_X = {}
    p2p_Y = {}

    print("Computing possible matchings for shape X and Y...")

    for i in range(vx.shape[0]):
        curr_key = vx2vx_org[i]
        if curr_key in p2pX_org:
            curr_p2p = np.unique(p2pX_org[curr_key])
            p2p_X[i] = np.nonzero(
                (vy2vy_org[None] == np.array(curr_p2p)[:, None]).any(axis=0)
            )[0]

    for i in range(vy.shape[0]):
        curr_key = vy2vy_org[i]
        if curr_key in p2pY_org:
            curr_p2p = np.unique(p2pY_org[curr_key])
            p2p_Y[i] = np.nonzero(
                (vx2vx_org[None] == np.array(curr_p2p)[:, None]).any(axis=0)
            )[0]

    print("Computing possible matchings done.")

    ex = igl.edges(fx)
    ex = np.row_stack((ex, ex[:, [1, 0]]))
    # triangles to cycles
    ey = get_surface_cycles(fy)

    ## Comptue Features and edge cost matrix
    feat_x = torch.load(feat_folder + shape_1 + "_feat.pt").squeeze(0).numpy()
    feat_y = torch.load(feat_folder + shape_2 + "_feat.pt").squeeze(0).numpy()
    overlap12 = (
        torch.load(feat_folder + shape_1 + "_" + shape_2 + "_overlap12.pt")
        .squeeze(0)
        .numpy()
    ).astype(np.float32)
    overlap21 = (
        torch.load(feat_folder + shape_1 + "_" + shape_2 + "_overlap21.pt")
        .squeeze(0)
        .numpy()
    ).astype(np.float32)
    overlap12_lowres = overlap12[vx2VX]
    overlap21_lowres = overlap21[vy2VY]

    edge_costs, _ = compute_edge_costs(feat_x, feat_y, vx, vx2VX, vy, vy2VY)

    pg = product_graph_generator(vx, ex, vy, ey, edge_costs)
    pg.set_resolve_coupling(resolve_coupling)
    pg.set_max_depth(max_depth)
    pg.generate()
    # c1,c2: indices in Y, c3,c4: indices in X, c5,c6: not interesting, c7: subproblem index (edes from ith sub product graph)
    product_space = pg.get_product_space()

    E = pg.get_cost_vector()
    # sparse matrix representation, I row, J column, V value
    I, J, V = pg.get_constraint_matrix_vectors()
    RHS = pg.get_rhs().flatten()

    # gurobi setup & solve
    m = gp.Model("surface_cycles")
    x = m.addMVar(shape=E.shape[0], vtype=GRB.BINARY, name="x")

    A = sp.csr_matrix(
        (V.flatten(), (I.flatten(), J.flatten())), shape=(RHS.shape[0], E.shape[0])
    )

    print("Start to prune product space...")
    pruned_product_space = get_pruned_search_space(vx, vy, p2p_X, p2p_Y, product_space)

    print("Pruning product space done.")

    # prune A, E and x
    E = E[pruned_product_space]
    x = x[pruned_product_space]
    product_space = product_space[pruned_product_space, :]

    # if needed P, L, S can be extracted from A
    P_index = np.array((np.abs(A).sum(axis=1))).flatten()
    P_index = np.logical_and(P_index > 2, P_index < A[-1, :].sum())
    P = A[P_index, :]
    L = A[np.array((np.abs(A).sum(axis=1)) == 2).flatten(), :]

    S = A[RHS == 1, :]

    P = P[:, pruned_product_space]
    L = L[:, pruned_product_space]
    S = S[:, pruned_product_space]

    mask_boundary_all = get_boundaries(fx, fy, product_space)

    m.addConstr(P @ x == 0, name="flow_conservation")
    m.addConstr(
        (L[:, ~mask_boundary_all] @ x[~mask_boundary_all]) == 0, name="coupling"
    )

    slack_S = m.addMVar(shape=S.shape[0], vtype=GRB.BINARY, name="slack_S")

    m.addConstr(S @ x + slack_S == 1, name="overlap_injectivity")
    overlap_mask = get_overlap_constraint_mask(product_space, S, fy, overlap21_lowres)

    slack_vx = m.addMVar(shape=vx.shape[0], vtype=GRB.BINARY, name="slack_vx")

    print("Adding constraints for overlap vertices in VX...")
    for i in range(vx.shape[0]):
        m.addConstr(
            x[np.any(product_space[:, [2, 3]] == i, axis=1)].sum() + slack_vx[i] >= 1,
            name=f"match at least one of v{i}",
        )

    print("Adding constraints for overlap vertices in VX done")
    cost = E.transpose()

    obj = (
        cost @ x
        + factor_slack_S * (overlap_mask @ slack_S)
        + factor_slack_vx * (overlap12_lowres @ slack_vx)
    )

    m.setObjective(obj, GRB.MINIMIZE)

    start_time = time.time()
    m.setParam("TimeLimit", time_limit)
    m.setParam("LogFile", os.path.join(curr_results_folder, "gurobi.log"))

    print("Starting optimization...")
    m.optimize()
    print("Optimisation done.")
    end_time = time.time()
    print(f"Optimisation took {end_time - start_time}s")
    # check if the model is infeasible
    if m.status == GRB.INFEASIBLE or m.status == GRB.TIME_LIMIT:
        print("Model is infeasible or time limit reached.")
    result_vec = x.X
    result_obj = obj.getValue()
    print("Objective value:", result_obj)
    print("Number of edges in matching:", np.sum(result_vec > 0.5))

    matching_edges = product_space[result_vec.astype("bool"), 0:4]

    p2p_y, p2p_x = result_to_p2p(result_vec, product_space, vx, vy)
    high_res_p2p_x = get_high_correspondences(p2p_x, VX, VX2vx, VY2vy)
    high_res_p2p_y = get_high_correspondences(p2p_y, VY, VY2vy, VX2vx)

    overlap_pred_x = high_res_p2p_x > -1
    overlap_pred_y = high_res_p2p_y > -1

    result_dict = {
        "result_vec": result_vec,
        "result_obj": result_obj,
        "vx": vx,
        "fx": fx,
        "vy": vy,
        "fy": fy,
        "VX": VX,
        "VY": VY,
        "FX": FX,
        "FY": FY,
        "overlap_pred_y": overlap_pred_y,
        "overlap_pred_x": overlap_pred_x,
        "matching_edges": matching_edges,
    }
    result_filename = os.path.join(curr_results_folder, "result.npz")
    np.savez(result_filename, **result_dict)

    vis_p2p(
        VY,
        VX,
        FY,
        FX,
        high_res_p2p_x,
        high_res_p2p_y,
        dataset=dataset,
        save_path=os.path.join(curr_results_folder, "matching_high_res.png"),
    )

    vis_p2p(
        vy,
        vx,
        fy,
        fx,
        p2p_x,
        p2p_y,
        dataset=dataset,
        save_path=os.path.join(curr_results_folder, "matching_low_res.png"),
    )
    # This function can be used to wrap the code in a function if needed


if __name__ == "__main__":
    map_filename = "dog-15_dog-6"
    coarse_to_fine(old_nr=600, new_nr=800, map_filename=map_filename, lamda=0.3)
    coarse_to_fine(old_nr=800, new_nr=1000, map_filename=map_filename, lamda=0.3)
