import numpy as np

from spyfe.fields.elemental_field import ElementalField


def is_node_in_element(node_xyz, element_xyzs, element_dim=1):
    """
    Checks if a node (node_xyz) is inside an element defined by element_xyzs.
    For 1D: checks if node is between two points in 2D space.
    For 2D: checks if node is inside a quadrilateral (convex hull).
    """
    node = np.array(node_xyz)
    element = np.array(element_xyzs)
    if element_dim == 1:
        # 1D element in 2D: check if node lies on the line segment
        a, b = element
        ab = b - a
        an = node - a
        cross = np.cross(ab, an)
        dot = np.dot(ab, an)
        if np.abs(cross) < 1e-8 and 0 <= dot <= np.dot(ab, ab):
            return True
        return False
    # elif element_dim == 2:
    #     path = Path(element)
    #     return path.contains_point(node)
    else:
        raise NotImplementedError("Only 1D elements are supported.")


def compute_xi(node_xyz, element_xyz, element_dim=1):
    """
    Compute the isoparametric coordinates (xi) of a node within an element.
    For 1D: maps node position to local coordinate in [-1, 1].
    For 2D: (not implemented).
    """
    node = np.array(node_xyz)
    element = np.array(element_xyz)
    if element_dim == 1:
        a, b = element
        # Map node position to local coordinate xi in [-1, 1]
        length = np.linalg.norm(b - a)
        if length == 0:
            raise ValueError("Element has zero length.")
        xi = 2 * np.dot(node - a, b - a) / np.dot(b - a, b - a) - 1
        return np.array([xi])
    else:
        raise NotImplementedError("Only 1D elements are supported.")


def assemble_gamma(subdomain_fens, subdomain_bfes, subdomain_interface_fe_idx, interface_fens):
    N = interface_fens.count()
    gamma = np.zeros((N, subdomain_fens.count()))
    for i in range(N):
        for elem in subdomain_bfes.conn[subdomain_interface_fe_idx]:
            elem_xyz = subdomain_fens.xyz[elem]
            node_xyz = interface_fens.xyz[i]
            if is_node_in_element(node_xyz, elem_xyz):
                # xi is isoparametric coordinate of node in terms of the element
                xi = compute_xi(node_xyz, elem_xyz)
                a = subdomain_bfes.bfun(xi).flatten()
                gamma[i, elem] += a
    # in case any entry is 2.0, it is because of the boundary nodes. this is a temporary fix.
    gamma[gamma == 2.0] = 1.0
    return gamma

def L2_err(femm, geom, temp, sol):
    err = ElementalField(fes = femm.fes)

    fes = femm.fes
    bfuns, gradbfunpars, npts, pc, w = femm.integration_data()

    mcs = femm.material_csys

    for i in range(fes.conn.shape[0]):
        x = geom.values[fes.conn[i, :], :]
        for j in range(npts):
            jacmat = np.dot(x.T, gradbfunpars[j])
            jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
            # err.values[i] += np.dot(gradbfun, np.dot((jac * w[j]) * kappa_bar, gradbfun.T))
            u = sol((np.array(bfuns)[j].T@x).flatten())
            uh = np.array(bfuns)[j].T@temp.values[fes.conn[i]].flatten()
            err.values[i] += (jac * w[j]) * (u-uh)**2
        err.values[i] = np.sqrt(err.values[i])
    return err


import numpy as np
from scipy.sparse import csr_matrix

def build_edge_map_simple(src_edge_xyz, tgt_xyz, tgt_conn, edge_elem_idx):
    """
    Build A such that b = A @ L, where:
      - L are piecewise constant values on the source edge elements
      - b are nodal loads on the target 2D mesh (P1)
    Only nodes on the specified edge contribute; others are zero.

    Parameters
    ----------
    src_edge_xyz : (m+1,2) array
        Coordinates of the source edge nodes (ordered along the edge).
    tgt_xyz : (n,2) array
        Coordinates of all target mesh nodes.
    tgt_conn : (n_edge_elems,2) int array
        Connectivity of all boundary elements (edges) in the 2D mesh.
    edge_elem_idx : 1D array of int
        Indices of the target boundary elements belonging to the interface edge.

    Returns
    -------
    A : (n_tgt_nodes, m_src_elems) csr_matrix
        Mapping matrix; b = A @ L
    """
    # Source arclength coordinate
    s_src = np.zeros(len(src_edge_xyz))
    s_src[1:] = np.cumsum(np.linalg.norm(np.diff(src_edge_xyz, axis=0), axis=1))
    src_breaks = s_src

    # Target edge coordinates and arclength
    edge_conn = tgt_conn[edge_elem_idx]
    edge_nodes = np.unique(edge_conn.flatten())
    edge_xyz = tgt_xyz[edge_nodes]
    s_tgt = np.zeros(len(edge_nodes))
    s_tgt[1:] = np.cumsum(np.linalg.norm(np.diff(edge_xyz, axis=0), axis=1))

    # Build 1D overlap operator (same as earlier)
    m = len(src_breaks) - 1
    n_edge = len(edge_nodes)
    A_edge = np.zeros((n_edge, m))
    for i in range(m):
        a, b = src_breaks[i], src_breaks[i+1]
        for j in range(n_edge - 1):
            xt0, xt1 = s_tgt[j], s_tgt[j+1]
            h = xt1 - xt0
            a_ = max(a, xt0); b_ = min(b, xt1)
            if b_ <= a_: continue
            A_edge[j, i]   += ((xt1*b_ - 0.5*b_**2) - (xt1*a_ - 0.5*a_**2)) / h
            A_edge[j+1, i] += ((0.5*b_**2 - xt0*b_) - (0.5*a_**2 - xt0*a_)) / h

    # Embed into full mesh
    A = np.zeros((tgt_xyz.shape[0], m))
    A[edge_nodes, :] = A_edge
    return csr_matrix(A)

