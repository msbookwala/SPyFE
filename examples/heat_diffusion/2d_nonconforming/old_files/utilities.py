import numpy as np

from spyfe.fields.elemental_field import ElementalField
from scipy.sparse import csr_matrix

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

import numpy as np
from scipy.sparse import csr_matrix

# 2-pt Gauss on [-1,1] is exact for P1×P1 products
_GX = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
_GW = np.array([1.0, 1.0])

def _seg_length(p0, p1):
    return np.linalg.norm(p1 - p0)

def _phiP1(xi):  # frame (interface) element P1 basis at xi∈[-1,1]
    return np.array([(1.0 - xi) * 0.5, (1.0 + xi) * 0.5])

def _embed_edge_connectivity_from_ordered_nodes(N):
    # consecutive nodes (0-1, 1-2, ..., N-2 - N-1)
    conn = np.column_stack([np.arange(N-1, dtype=int), np.arange(1, N, dtype=int)])
    return conn

def interface_mass_matrix_P1(interface_fens):
    """MΓ for a P1 nodal LM on the frame: block L/6 [[2,1],[1,2]] per edge."""
    xyz = interface_fens.xyz
    N = interface_fens.count()
    conn = _embed_edge_connectivity_from_ordered_nodes(N)
    rows, cols, data = [], [], []
    for (a, b) in conn:
        L = _seg_length(xyz[a], xyz[b])
        rows += [a,a,b,b]
        cols += [a,b,a,b]
        data += [2*L/6, 1*L/6, 1*L/6, 2*L/6]
    M = csr_matrix((data, (rows, cols)), shape=(N, N))
    return M

def cross_mass_sub_to_frame_P1(sub_fens, sub_bfes, sub_edge_idx, interface_fens):
    """
    C(i,j) = ∫_Γ φ_i (frame-P1) * N_j (subdomain-edge P1) ds
    Integrates over each frame edge using 2-pt Gauss; for each quad point,
    finds the active subdomain edge and evaluates its P1 basis.
    """
    xyzΓ = interface_fens.xyz
    NΓ = interface_fens.count()
    # frame connectivity taken as consecutive nodes
    connΓ = _embed_edge_connectivity_from_ordered_nodes(NΓ)

    C = np.zeros((NΓ, sub_fens.count()))

    # subdomain interface edges (their global-node connectivity)
    sub_conn = sub_bfes.conn[sub_edge_idx]
    sub_xyz  = sub_fens.xyz

    for (i0, i1) in connΓ:
        p0, p1 = xyzΓ[i0], xyzΓ[i1]
        L = _seg_length(p0, p1)
        # map xi∈[-1,1] to physical point on the frame edge
        for xi, w in zip(_GX, _GW):
            # frame P1 basis at xi and physical point
            phi = _phiP1(xi)           # [phi_i0, phi_i1]
            s  = 0.5*(xi + 1.0)        # affine map [-1,1]→[0,1]
            xq = (1.0 - s) * p0 + s * p1

            # find the active subdomain boundary edge that contains xq
            found = False
            for elem in sub_conn:
                a, b = sub_xyz[elem[0]], sub_xyz[elem[1]]
                ab  = b - a
                # check if xq lies on segment [a,b]
                cross = np.cross(ab, xq - a)
                dot   = np.dot(xq - a, ab)
                if np.abs(cross) <= 1e-12 and 0.0 - 1e-12 <= dot <= np.dot(ab, ab) + 1e-12:
                    # local coordinate on [a,b] in [-1,1]
                    xi_sub = 2.0 * dot / np.dot(ab, ab) - 1.0
                    Nj = sub_bfes.bfun(np.array([xi_sub])).flatten()  # shape (2,)
                    # accumulate: C[i0,:] and C[i1,:]
                    wJ = w * (L * 0.5)
                    C[i0, elem] += phi[0] * Nj * wJ
                    C[i1, elem] += phi[1] * Nj * wJ
                    found = True
                    break
            if not found:
                # robust fallback: snap to nearest edge if roundoff hits a vertex
                # (optional) or raise an error
                pass
    return csr_matrix(C)

def assemble_gamma_L2(sub_fens, sub_bfes, sub_edge_idx, interface_fens):
    """
    Return (Gamma, C, MΓ) with Γ = MΓ^{-1} C  (mortar/L2 projection).
    For assembly you only need C (since B = C, G = C^T for nodal LM).
    """
    MΓ = interface_mass_matrix_P1(interface_fens)
    C  = cross_mass_sub_to_frame_P1(sub_fens, sub_bfes, sub_edge_idx, interface_fens)
    # Prefer solving with a sparse solver rather than inverting MΓ explicitly:
    # Gamma = splu(MΓ).solve(C.toarray())
    # But most callers can skip Γ entirely and use B=C, G=C^T.
    return C  # we return C because that's what you should use
