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


import os

from vtkmodules.vtkCommonDataModel import (
    vtkDataObject, vtkDataSet, vtkPolyData, vtkUnstructuredGrid,
    vtkImageData, vtkRectilinearGrid, vtkStructuredGrid
)
from vtkmodules.vtkIOLegacy import vtkDataSetReader, vtkDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter


# -------------------- I/O helpers --------------------
def _read_any(path: str) -> vtkDataObject:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    # XML reader (usually available in minimal builds)
    if ext.startswith(".vt") or ext.startswith(".pv"):
        r = vtkXMLGenericDataObjectReader()
        r.SetFileName(path)
        r.Update()
        out = r.GetOutputDataObject(0)
        if out:
            return out

    # Legacy .vtk fallback
    if ext == ".vtk":
        r = vtkDataSetReader()
        r.SetFileName(path)
        r.Update()
        out = r.GetOutput()
        if out:
            return out

    raise RuntimeError(f"Unsupported or unreadable file: {path}")


def _write_any(obj: vtkDataObject, filename: str):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".vtk":
        if not isinstance(obj, vtkDataSet):
            raise RuntimeError("Writing .vtk requires a vtkDataSet.")
        w = vtkDataSetWriter()
        w.SetFileName(filename)
        w.SetInputData(obj)
        if w.Write() == 0:
            raise RuntimeError(f"Failed to write {filename}")
        return

    # XML writers: import lazily to avoid missing symbols in trimmed builds
    if obj.IsA("vtkUnstructuredGrid") and ext == ".vtu":
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
        w = vtkXMLUnstructuredGridWriter()
    elif obj.IsA("vtkPolyData") and ext == ".vtp":
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
        w = vtkXMLPolyDataWriter()
    elif obj.IsA("vtkImageData") and ext == ".vti":
        from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
        w = vtkXMLImageDataWriter()
    elif obj.IsA("vtkStructuredGrid") and ext == ".vts":
        from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter
        w = vtkXMLStructuredGridWriter()
    elif obj.IsA("vtkRectilinearGrid") and ext == ".vtr":
        from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
        w = vtkXMLRectilinearGridWriter()
    else:
        # Common case after AppendFilter is vtkUnstructuredGrid:
        # if the extension doesn't match, raise with a helpful hint
        raise RuntimeError(
            f"Unsupported dataset/extension combo: {obj.GetClassName()} -> {filename}\n"
            f"If you used AppendFilter, the result is likely vtkUnstructuredGrid. Use a .vtu extension."
        )

    w.SetFileName(filename)
    w.SetInputData(obj)
    if hasattr(w, "SetCompressorTypeToZLib"):
        w.SetCompressorTypeToZLib()
    if w.Write() == 0:
        raise RuntimeError(f"Failed to write {filename}")


# -------------------- field utilities --------------------
def _attrs(ds: vtkDataSet, assoc: str):
    return ds.GetPointData() if assoc == "POINTS" else ds.GetCellData()

def _names_and_ncomps(ds: vtkDataSet, assoc: str) -> dict:
    cont = _attrs(ds, assoc)
    if not cont:
        return {}
    out = {}
    for i in range(cont.GetNumberOfArrays()):
        arr = cont.GetArray(i)
        if not arr:
            continue
        nm = arr.GetName() or ""
        if nm:
            out[nm] = arr.GetNumberOfComponents()
    return out

def _keep_only(ds: vtkDataSet, assoc: str, names_to_keep: set[str]):
    cont = _attrs(ds, assoc)
    if not cont:
        return
    to_remove = []
    for i in range(cont.GetNumberOfArrays()):
        arr = cont.GetArray(i)
        nm = arr.GetName() if arr else None
        if nm and nm not in names_to_keep:
            to_remove.append(nm)
    for nm in to_remove:
        cont.RemoveArray(nm)


# -------------------- merge core --------------------
def _append_two_to_ugrid(a: vtkDataSet, b: vtkDataSet) -> vtkUnstructuredGrid:
    app = vtkAppendFilter()
    app.AddInputData(a)
    app.AddInputData(b)
    app.Update()
    out = app.GetOutput()
    # out is vtkUnstructuredGrid
    return out


def merge_vtk_files_common_fields(
    file_a: str,
    file_b: str,
    output_path: str | None = None,
    include_field_data: bool = False,  # FIELD data not commonly needed for coloring
):
    A = _read_any(file_a)
    B = _read_any(file_b)

    if not isinstance(A, vtkDataSet) or not isinstance(B, vtkDataSet):
        raise RuntimeError("Both inputs must be vtkDataSet (e.g., .vtu, .vtp, .vts, .vtr, .vti, .vtk).")

    # POINT arrays: keep intersection with matching ncomp
    namesA_pts = _names_and_ncomps(A, "POINTS")
    namesB_pts = _names_and_ncomps(B, "POINTS")
    common_pts = {n for n in namesA_pts.keys() & namesB_pts.keys() if namesA_pts[n] == namesB_pts[n]}
    _keep_only(A, "POINTS", common_pts)
    _keep_only(B, "POINTS", common_pts)

    # CELL arrays: same deal
    namesA_cls = _names_and_ncomps(A, "CELLS")
    namesB_cls = _names_and_ncomps(B, "CELLS")
    common_cls = {n for n in namesA_cls.keys() & namesB_cls.keys() if namesA_cls[n] == namesB_cls[n]}
    _keep_only(A, "CELLS", common_cls)
    _keep_only(B, "CELLS", common_cls)

    if include_field_data:
        contA = A.GetFieldData()
        contB = B.GetFieldData()
        def names_fd(ds):
            if not ds.GetFieldData(): return {}
            out = {}
            cd = ds.GetFieldData()
            for i in range(cd.GetNumberOfArrays()):
                arr = cd.GetArray(i)
                if arr and arr.GetName():
                    out[arr.GetName()] = arr.GetNumberOfComponents()
            return out
        def keep_fd(ds, keep):
            cd = ds.GetFieldData()
            if not cd: return
            to_remove = []
            for i in range(cd.GetNumberOfArrays()):
                arr = cd.GetArray(i)
                nm = arr.GetName() if arr else None
                if nm and nm not in keep:
                    to_remove.append(nm)
            for nm in to_remove:
                cd.RemoveArray(nm)
        namesA_fd = names_fd(A)
        namesB_fd = names_fd(B)
        common_fd = {n for n in namesA_fd.keys() & namesB_fd.keys()
                     if namesA_fd[n] == namesB_fd[n]}
        keep_fd(A, common_fd)
        keep_fd(B, common_fd)

    merged = _append_two_to_ugrid(A, B)

    if output_path:
        _write_any(merged, output_path)

    return merged
########################################################################################################################

def _arclength_nodes(xyz):
    s = np.zeros(len(xyz))
    if len(xyz) > 1:
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        s[1:] = np.cumsum(d)
    return s

def _lin_coeffs_on_segment(sL, sR, which):
    h = (sR - sL)
    if which == 0:   # (sR - s)/h
        return (sR / h, -1.0 / h)
    else:            # (s - sL)/h
        return (-sL / h, 1.0 / h)

def _int_linlin(a0,a1,b0,b1, sA, sB):
    A = a0*b0
    B = a0*b1 + a1*b0
    C = a1*b1
    return A*(sB - sA) + 0.5*B*(sB**2 - sA**2) + (1.0/3.0)*C*(sB**3 - sA**3)

def cross_mass_P1_frame_P1_sub(frame_xyz, sub_xyz, sub_edge_conn):
    # Order subdomain interface nodes
    edge_nodes = _order_edge_nodes(sub_edge_conn)
    sub_edge_xyz = sub_xyz[edge_nodes]
    sF = _arclength_nodes(frame_xyz)
    sS = _arclength_nodes(sub_edge_xyz)

    # Breakpoints of the union partition
    brk = np.union1d(sF, sS)

    C = np.zeros((len(frame_xyz), len(edge_nodes)))
    # Pointers to current active elements
    k = 0  # frame element index (between nodes k and k+1)
    l = 0  # sub edge element index (between nodes l and l+1)

    for i in range(len(brk) - 1):
        a, b = brk[i], brk[i+1]
        # advance k,l to the element containing [a,b]
        while not (sF[k] <= a + 1e-14 and b <= sF[k+1] + 1e-14):
            k += 1
        while not (sS[l] <= a + 1e-14 and b <= sS[l+1] + 1e-14):
            l += 1
        # frame bases on [sF[k], sF[k+1]]
        fL_a0, fL_a1 = _lin_coeffs_on_segment(sF[k],   sF[k+1], which=0)
        fR_a0, fR_a1 = _lin_coeffs_on_segment(sF[k],   sF[k+1], which=1)
        # sub bases on [sS[l], sS[l+1]]
        sL_b0, sL_b1 = _lin_coeffs_on_segment(sS[l],   sS[l+1], which=0)
        sR_b0, sR_b1 = _lin_coeffs_on_segment(sS[l],   sS[l+1], which=1)

        C[k,   l]   += _int_linlin(fL_a0,fL_a1, sL_b0,sL_b1, a, b)
        C[k,   l+1] += _int_linlin(fL_a0,fL_a1, sR_b0,sR_b1, a, b)
        C[k+1, l]   += _int_linlin(fR_a0,fR_a1, sL_b0,sL_b1, a, b)
        C[k+1, l+1] += _int_linlin(fR_a0,fR_a1, sR_b0,sR_b1, a, b)

    return C, edge_nodes

def embed_cross_mass_to_full(C_edge, edge_nodes, n_total_nodes):
    rows, cols, data = [], [], []
    for a in range(C_edge.shape[0]):
        nz = np.nonzero(C_edge[a, :])[0]
        rows.extend([a]*len(nz))
        cols.extend(edge_nodes[nz])
        data.extend(C_edge[a, nz])
    return csr_matrix((data, (rows, cols)), shape=(C_edge.shape[0], n_total_nodes))

import numpy as np
from scipy.sparse import csr_matrix

def _order_edge_nodes(edge_conn):
    # Build adjacency
    adj = {}
    for a, b in edge_conn:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # Find an endpoint (degree 1), else start anywhere
    ends = [n for n, nb in adj.items() if len(nb) == 1]
    start = ends[0] if ends else edge_conn[0, 0]
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(adj):  # safety
            break
    return np.array(order, dtype=int)



################

def p0p1_order_edge_nodes_chain__no_reuse(edge_conn: np.ndarray) -> np.ndarray:
    adj = {}
    for a, b in edge_conn:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    ends = [n for n, nb in adj.items() if len(nb) == 1]
    start = ends[0] if ends else edge_conn[0, 0]
    order, prev, cur = [start], -1, start
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(adj):  # safety
            break
    return np.array(order, dtype=int)

def p0p1_arclength_param__no_reuse(xyz: np.ndarray) -> np.ndarray:
    s = np.zeros(len(xyz))
    if len(xyz) > 1:
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        s[1:] = np.cumsum(d)
    return s

def p0p1_lin_coeffs_on_segment__no_reuse(sL: float, sR: float, which: int) -> tuple[float, float]:
    h = (sR - sL)
    if h <= 0:
        return (0.0, 0.0)
    if which == 0:   # left-node basis (sR - s)/h
        return (sR / h, -1.0 / h)
    else:            # right-node basis (s - sL)/h
        return (-sL / h, 1.0 / h)

def p0p1_int_lin__no_reuse(a0: float, a1: float, A: float, B: float) -> float:
    return a0 * (B - A) + 0.5 * a1 * (B**2 - A**2)

def p0p1_crossmass_edge__no_reuse(src_frame_xyz: np.ndarray,
                                  tgt_edge_xyz: np.ndarray,
                                  tol: float = 1e-13) -> np.ndarray:

    sF = p0p1_arclength_param__no_reuse(src_frame_xyz)     # frame arclength nodes
    sS = p0p1_arclength_param__no_reuse(tgt_edge_xyz)      # sub-edge arclength nodes
    NL = max(0, len(sF) - 1)                               # #frame elements
    if NL == 0 or len(sS) < 2:
        return np.zeros((0, len(tgt_edge_xyz)))

    brk = np.union1d(sF, sS)
    s_min, s_max = brk[0], brk[-1]
    D_edge = np.zeros((NL, len(tgt_edge_xyz)))

    for i in range(len(brk) - 1):
        A, B = brk[i], brk[i + 1]
        if B - A <= tol:
            continue
        # midpoint strictly inside [A,B) to pick active elements
        mid = 0.5 * (A + B)
        mid = min(max(mid, s_min + tol), s_max - tol)

        # active frame element k: sF[k] <= mid < sF[k+1]
        k = np.searchsorted(sF, mid, side='right') - 1
        k = max(0, min(k, NL - 1))

        # active sub-edge element l: sS[l] <= mid < sS[l+1]
        l = np.searchsorted(sS, mid, side='right') - 1
        l = max(0, min(l, len(sS) - 2))  # ensure l+1 exists

        # sub P1 bases as linear polynomials on [sS[l], sS[l+1]]
        Nl_a0, Nl_a1 = p0p1_lin_coeffs_on_segment__no_reuse(sS[l],   sS[l + 1], which=0)
        Nr_a0, Nr_a1 = p0p1_lin_coeffs_on_segment__no_reuse(sS[l],   sS[l + 1], which=1)

        # clip overlap to avoid FP drift at endpoints
        AA, BB = max(A, sS[l]), min(B, sS[l + 1] - tol)
        if BB - AA <= tol:
            continue

        # χ_e = 1 on frame element k
        D_edge[k, l]     += p0p1_int_lin__no_reuse(Nl_a0, Nl_a1, AA, BB)
        D_edge[k, l + 1] += p0p1_int_lin__no_reuse(Nr_a0, Nr_a1, AA, BB)

    return D_edge  # shape: NL × n_edge_nodes

def p0p1_embed_edgecols_to_full__no_reuse(D_edge: np.ndarray,
                                          edge_nodes: np.ndarray,
                                          n_total_nodes: int) -> csr_matrix:
    rows, cols, data = [], [], []
    NL, n_edge_nodes = D_edge.shape
    for e in range(NL):
        nz = np.nonzero(D_edge[e, :])[0]
        rows.extend([e] * len(nz))
        cols.extend(edge_nodes[nz])
        data.extend(D_edge[e, nz])
    return csr_matrix((data, (rows, cols)), shape=(NL, n_total_nodes))

def build_p0p1_interpolator_frame_to_sub_full__no_reuse(src_frame_xyz: np.ndarray,
                                                        tgt_xyz: np.ndarray,
                                                        tgt_conn: np.ndarray,
                                                        edge_elem_idx: np.ndarray) -> csr_matrix:

    # order the subdomain interface nodes from the provided boundary connectivity
    edge_conn_subset = tgt_conn[edge_elem_idx,:]  # interface edges only
    edge_nodes_ordered = p0p1_order_edge_nodes_chain__no_reuse(edge_conn_subset)
    tgt_edge_xyz = tgt_xyz[edge_nodes_ordered]

    # build edge-only cross-mass then embed to full NT
    D_edge = p0p1_crossmass_edge__no_reuse(src_frame_xyz, tgt_edge_xyz)   # NL × n_edge_nodes
    D_full = p0p1_embed_edgecols_to_full__no_reuse(D_edge, edge_nodes_ordered, tgt_xyz.shape[0])
    return D_full  # NL × NT (CSR)
