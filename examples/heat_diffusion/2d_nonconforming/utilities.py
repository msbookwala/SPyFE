import numpy as np

from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.fields.elemental_field import ElementalField
from scipy.sparse import csr_matrix
import os

from vtkmodules.vtkCommonDataModel import (
    vtkDataObject, vtkDataSet, vtkPolyData, vtkUnstructuredGrid,
    vtkImageData, vtkRectilinearGrid, vtkStructuredGrid
)
from vtkmodules.vtkIOLegacy import vtkDataSetReader, vtkDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter

from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.meshing.generators.intervals import l2_blockx_2D


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
        if np.abs(cross) < 1e-8 and 0 <= dot <= np.dot(ab, ab)+1e-8:
            return True
        return False
    # elif element_dim == 2:
    #     path = Path(element)
    #     return path.contains_point(node)
    else:
        raise NotImplementedError("Only 1D elements are supported.")


def compute_xi(node_xyz, element_xyz, element_dim=1):
    node = np.asarray(node_xyz, dtype=np.float64)
    element = np.asarray(element_xyz, dtype=np.float64)
    if element_dim == 1:
        a = element[0]; b = element[1]
        v = b - a
        denom = np.dot(v, v)
        if denom == 0.0:
            raise ValueError("Element has zero length.")
        xi = 2.0 * np.dot(node - a, v) / denom - 1.0
        # clamp tiny overshoot due to rounding
        if xi < -1.0 - 1e-14: xi = -1.0
        if xi >  1.0 + 1e-14: xi =  1.0
        return float(xi)
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

def lagrange_interpolation_matrix(nodes1, nodes2, atol=1e-12):
    nodes1 = np.asarray(nodes1, dtype=np.float64)
    nodes2 = np.asarray(nodes2, dtype=np.float64)
    N1, N2 = nodes1.shape[0], nodes2.shape[0]
    M = np.zeros((N2, N1), dtype=np.float64)
    if N1 == 0: return M
    if N1 == 1:
        M[:, 0] = 1.0
        return M

    for j in range(N2):
        p = nodes2[j, :]
        # exact match
        d = np.linalg.norm(nodes1 - p, axis=1)
        k_min = int(np.argmin(d))
        if d[k_min] <= atol:
            M[j, k_min] = 1.0
            continue

        placed = False
        for i in range(N1 - 1):
            a = nodes1[i, :]
            b = nodes1[i + 1, :]
            if is_node_in_element(p, [a, b]):
                xi = compute_xi(p, [a, b])   # scalar now
                # linear shape functions
                N_left  = 0.5 * (1.0 - xi)
                N_right = 0.5 * (1.0 + xi)
                # numerical safety: clamp small negative zeros
                if N_left < 1e-16: N_left = 0.0
                if N_right < 1e-16: N_right = 0.0
                M[j, i]     += float(N_left)
                M[j, i + 1] += float(N_right)
                placed = True
                break
        if not placed:
            # robust fallback: nearest neighbor
            M[j, k_min] = 1.0
    return M


def pwc_interpolation_matrix(nodes1, nodes2, atol=1e-12):
    nodes1 = np.asarray(nodes1, dtype=float)
    nodes2 = np.asarray(nodes2, dtype=float)

    midpts = (nodes2[:-1] + nodes2[1:]) / 2
    mat = np.zeros((nodes2.shape[0]-1, nodes1.shape[0]-1))

    for i in range(nodes2.shape[0]-1):
        for j in range(nodes1.shape[0]-1):
            a = nodes1[j]
            b = nodes1[j+1]
            if is_node_in_element(midpts[i], [a, b]):
                mat[i, j] = 1.0
    return mat

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

def flux(femm, geom, temp, sol):
    flux = ElementalField(nelems = femm.fes.conn.shape[0], dim = 2)

    fes = femm.fes
    bfuns, gradbfunpars, npts, pc, w = femm.integration_data()

    mcs = femm.material_csys

    for i in range(fes.conn.shape[0]):
        x = geom.values[fes.conn[i, :], :]
        for j in range(npts):
            jacmat = np.dot(x.T, gradbfunpars[j])
            jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
            # flux.values[i] += np.dot(gradbfun, np.dot((jac * w[j]) * kappa_bar, gradbfun.T))
            du = np.array(gradbfunpars)[j].T@temp.values[fes.conn[i]].flatten()
            flux.values[i,:] += (jac * w[j]) * du
    return flux

def build_edge_map_simple(src_edge_xyz, tgt_xyz, tgt_conn, edge_elem_idx):
    # PO frame ->P1 subdomaing edge . flux to load

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


########################################################################################################################
# Merging VTK#
########################################################################################################################
def _read_any(path):
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


def _write_any(obj, filename):
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


def _attrs(ds, assoc):
    return ds.GetPointData() if assoc == "POINTS" else ds.GetCellData()

def _names_and_ncomps(ds, assoc):
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

def _keep_only(ds, assoc, names_to_keep):
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


def _append_two_to_ugrid(a, b):
    app = vtkAppendFilter()
    app.AddInputData(a)
    app.AddInputData(b)
    app.Update()
    out = app.GetOutput()
    # out is vtkUnstructuredGrid
    return out


def merge_vtk_files_common_fields(file_a, file_b, output_path, include_field_data=False):
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
# interpolation matrices
########################################################################################################################
def _order_edge_nodes(edge_conn):
    """Order a set of 2-node boundary edges into a single node chain."""
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

def build_interface_interpolator(frame_xyz, tgt_xyz, tgt_conn, edge_elem_idx, elem_lagrange = True, tol = 1e-13, give_both =False) :

    if elem_lagrange:
        lm_degree = "p0"
    else:
        lm_degree = "p1"
    # Order target interface nodes
    edge_conn = tgt_conn[edge_elem_idx]
    edge_nodes = _order_edge_nodes(edge_conn)
    edge_xyz   = tgt_xyz[edge_nodes]
    edge_xyz_, frame_xyz_ = trim(edge_xyz, frame_xyz, tol=tol)



    xys = unique_points_tol(np.vstack([frame_xyz_, edge_xyz_]), tol=1e-13)
    ys_i = xys[:,1]
    xs_i = xys[:,0]
    fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)



    geom_i = NodalField(fens=fens_i)
    if lm_degree.lower() == "p0":
        mu = ElementalField(nelems=fes_i.count(), dim=1)
        mu.numberdofs()
    else:
        mu =  NodalField(nfens=fens_i.count(), dim=1)
        mu.numberdofs()
    m = MatHeatDiff(thermal_conductivity=np.array([[1, 0.0], [0.0, 1]]), rho=1.0)
    femm_i = FEMMHeatDiff(fes=fes_i, material=m, integration_rule=GaussRule(dim=1, order=3))

    if lm_degree.lower() == "p0":
        A_ = lagrange_interpolation_matrix(edge_xyz, xys)
        B_ = pwc_interpolation_matrix(frame_xyz, xys)
        M = femm_i.lam_mat(geom_i, mu)
        M_edge = B_.T@M@A_
    else:
        A_ = lagrange_interpolation_matrix(edge_xyz, xys)
        B_ = lagrange_interpolation_matrix(frame_xyz, xys)
        M = femm_i.mass(geom_i, mu)
        M_edge = B_.T @ M @ A_

    # Embed edge-only columns into full NT
    rows, cols, data = [], [], []
    for r in range(M_edge.shape[0]):
        nz = np.nonzero(M_edge[r, :])[0]
        rows.extend([r]*len(nz))
        cols.extend(edge_nodes[nz])
        data.extend(M_edge[r, nz])
    if give_both:
        return csr_matrix((data, (rows,  cols)), shape=(M_edge.shape[0], tgt_xyz.shape[0])), M_edge
    else:
        return csr_matrix((data, (rows,  cols)), shape=(M_edge.shape[0], tgt_xyz.shape[0]))


def unique_points_tol(pts, tol=1e-12):
    pts = np.asarray(pts, dtype=np.float64)
    # sort by coordinates to make deterministic
    idx = np.lexsort((pts[:,1], pts[:,0]))
    pts_s = pts[idx]
    keep = []
    last = None
    for p in pts_s:
        if last is None or np.linalg.norm(p - last) > tol:
            keep.append(tuple(p))
            last = p
    return np.asarray(keep, dtype=np.float64)

def trim(edgexyz, frame_xyz, tol=1e-13):
    kept_edge = []
    kept_frame = []

    endpts_frame = [frame_xyz[0], frame_xyz[-1]]
    endpts_edge = [edgexyz[0], edgexyz[-1]]


    for p in edgexyz:
        if is_node_in_element(p, endpts_frame):
            kept_edge.append(p)
    for p in frame_xyz:
        if is_node_in_element(p, endpts_edge):
            kept_frame.append(p)
    return np.array(kept_edge, dtype=np.float64), kept_frame


