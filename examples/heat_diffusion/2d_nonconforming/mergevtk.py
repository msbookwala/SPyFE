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
