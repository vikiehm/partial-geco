import gurobipy as gp
from gurobipy import GRB
import time
import scipy.sparse as sp
from geco import product_graph_generator, get_surface_cycles
import polyscope as ps
import igl
import numpy as np
import os
from pathlib import Path
import torch

from utils.vis_utils import (
    vis_p2p,
)
from utils.shape_util import (
    compute_edge_costs,
    shape_loader,
    compute_geodesic_dist_matrices,
)
from utils.p2p_geco_utils import (
    get_boundaries,
    get_overlap_constraint_mask,
    result_to_p2p,
    get_high_correspondences,
)

ps.set_allow_headless_backends(True)


def solve_partial_geco(map_filename, lamda=0.3, dataset="cp2p", num_faces=600):
    time_limit = 3600  # 60 minutes
    max_depth = 2  # multiples of 2, quadratic size increase
    resolve_coupling = False  # makes the opt problem smaller
    print("Using number of faces:", num_faces)

    factor_slack_S = lamda
    factor_slack_vx = lamda

    filename1 = f"{map_filename.split('_')[0]}.off"
    filename2 = f"{map_filename.split('_')[1]}.off"
    off_folder = Path("./example_data/shapes/")

    basename1 = filename1.split(".")[0]
    basename2 = filename2.split(".")[0]

    filename1 = os.path.join(off_folder, filename1)
    filename2 = os.path.join(off_folder, filename2)

    results_folder = f"./results/{dataset}_{num_faces}/"
    dist_path = Path("./geo_matrices/")
    feat_folder = Path("./example_data/features/")

    print("Using files:", filename1, filename2)

    feat_x = torch.load(feat_folder / (basename1 + "_feat.pt")).squeeze(0).numpy()
    feat_y = torch.load(feat_folder / (basename2 + "_feat.pt")).squeeze(0).numpy()
    overlap12 = (
        torch.load(feat_folder / (basename1 + "_" + basename2 + "_overlap12.pt"))
        .squeeze(0)
        .numpy()
    ).astype(np.float32)
    overlap21 = (
        torch.load(feat_folder / (basename1 + "_" + basename2 + "_overlap21.pt"))
        .squeeze(0)
        .numpy()
    ).astype(np.float32)

    shape_opts = {
        "num_faces": num_faces,
        "partial_cp2p": True,
        "ratio_overlap_12": overlap12.sum() / overlap12.size,
        "ratio_overlap_21": overlap21.sum() / overlap21.size,
        "geodist_path": dist_path,
    }
    # generate results folder if it does not exist
    curr_results_folder = os.path.join(results_folder, map_filename)
    if not os.path.exists(curr_results_folder):
        os.makedirs(curr_results_folder)

    ## Load and downsample shapes and compute spidercurve on shape X
    VX, FX, vx, fx, vx2VX, VX2vx, VY, FY, vy, fy, vy2VY, VY2vy = shape_loader(
        filename1, filename2, shape_opts
    )
    ex = igl.edges(fx)
    ex = np.row_stack((ex, ex[:, [1, 0]]))
    # triangles to cycles
    ey = get_surface_cycles(fy)
    DX, DY = compute_geodesic_dist_matrices(
        dist_path, basename1, basename2, VX, FX, VY, FY
    )

    overlap12_lowres = overlap12[vx2VX]
    overlap21_lowres = overlap21[vy2VY]

    edge_costs, _ = compute_edge_costs(
        feat_x,
        feat_y,
        vx,
        vx2VX,
        vy,
        vy2VY,
        DX=DX,
        DY=DY,
    )
    ## ++++++++++++++++++++++++++++++++++++++++
    ## +++++++ Solve with SurfaceCycles +++++++
    ## ++++++++++++++++++++++++++++++++++++++++
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

    # if needed P, L, S can be extracted from A
    P_index = np.array((np.abs(A).sum(axis=1))).flatten()
    P_index = np.logical_and(P_index > 2, P_index < A[-1, :].sum())
    P = A[P_index, :]
    L = A[np.array((np.abs(A).sum(axis=1)) == 2).flatten(), :]

    S = A[RHS == 1, :]

    mask_boundary_all = get_boundaries(fx, fy, product_space)

    m.addConstr(P @ x == 0, name="flow_conservation")
    m.addConstr(
        (L[:, ~mask_boundary_all] @ x[~mask_boundary_all]) == 0, name="coupling"
    )

    overlap_mask = get_overlap_constraint_mask(product_space, S, fy, overlap21_lowres)

    slack_S = m.addMVar(shape=S.shape[0], vtype=GRB.BINARY, name="slack_S")
    m.addConstr(S @ x + slack_S == 1, name="injectivity")

    slack_vx = m.addMVar(shape=vx.shape[0], vtype=GRB.BINARY, name="slack_vx")

    for i in range(vx.shape[0]):
        # if i in overlap_vertices_vx:
        m.addConstr(
            x[np.any(product_space[:, [2, 3]] == i, axis=1)].sum() + slack_vx[i] >= 1,
            name=f"match at least one of v{i}",
        )

    cost = E.transpose()

    obj = (
        cost @ x
        + factor_slack_S * (overlap_mask @ slack_S)
        + factor_slack_vx * (overlap12_lowres @ slack_vx)
    )

    m.setObjective(obj, GRB.MINIMIZE)

    m.setParam("TimeLimit", time_limit)
    # write stdout to file
    m.setParam("LogFile", os.path.join(curr_results_folder, "gurobi.log"))
    start_time = time.time()
    m.optimize()
    end_time = time.time()
    print(f"Optimisation took {end_time - start_time}s")
    # check if the model is infeasible
    if m.status == GRB.INFEASIBLE or m.status == GRB.TIME_LIMIT:
        print("Model is infeasible or time limit reached.")
        # continue
    result_vec = x.X
    result_obj = obj.getValue()
    print("Objective value:", result_obj)
    print("Number of edges in matching:", np.sum(result_vec > 0.5))

    result_filename = os.path.join(curr_results_folder, "result.npz")

    matching_edges = product_space[result_vec.astype("bool"), 0:4]

    p2p_y, p2p_x = result_to_p2p(result_vec, product_space, vx, vy)
    high_res_p2p_x = get_high_correspondences(p2p_x, VX, VX2vx, VY2vy)
    high_res_p2p_y = get_high_correspondences(p2p_y, VY, VY2vy, VX2vx)

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

    overlap_pred_x = high_res_p2p_x > -1
    overlap_pred_y = high_res_p2p_y > -1

    result_dict = {
        "result_vec": result_vec,
        "result_obj": result_obj,
        "matching_edges": matching_edges,
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
    }
    np.savez(result_filename, **result_dict)


if __name__ == "__main__":
    solve_partial_geco()
