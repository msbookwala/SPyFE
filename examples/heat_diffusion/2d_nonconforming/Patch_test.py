import os
import sys
from scipy.special import lmbda
from spyfe.fields.elemental_field import ElementalField
from spyfe.meshing.boxes import bounding_box
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.quadrilaterals import q4_blockx
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
import numpy as np
from numpy import array
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.femms.femm_defor import FEMMDefor
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule, TriRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport
from spyfe.meshing.generators.intervals import l2_blockx_2D
from spyfe.meshing.selection import connected_nodes, fe_select, fenode_select
from matplotlib.path import Path
from scipy.sparse import bmat
from utilities import *
import pyvista as pv
from spyfe.meshing.generators.triangles import t3_ablock
from scipy.integrate import trapezoid



N_elem1 = 2
N_elem2 = 3
N_elem_i = min(N_elem1, N_elem2)
left_m = "quad"
right_m = "tri"

k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]), rho=1.0)
########################################################################################################################
# Side left
########################################################################################################################
if left_m == "quad":
    xs1 = np.linspace(0.0, 1.0, int(N_elem1/2)+1)
    ys1 = np.linspace(0.0, 2.0, N_elem1+1)
    fens1, fes1 = q4_blockx(xs1, ys1)
    femm1 = FEMMHeatDiff(fes = fes1, material=m, integration_rule=GaussRule(dim=2, order=2))
else:
    fens1, fes1 = t3_ablock(1, 2, int(N_elem1/2), N_elem1)
    femm1 = FEMMHeatDiff(fes = fes1, material=m, integration_rule=TriRule(npts=1))

T1 = NodalField(nfens=fens1.count(), dim=1)
geom1 = NodalField(fens=fens1)
T1.numberdofs()
K1 = femm1.conductivity(geom1, T1)
F1=femm1.nz_ebc_loads_conductivity(geom1, T1)
boundary_fes1 = mesh_boundary(fes1)
femm_left = FEMMHeatDiff(fes = boundary_fes1, material=m, integration_rule=GaussRule(dim=1, order=2))
fi_1 = ForceIntensity(magn=lambda x, J: -1.0 if np.isclose(x[0], 0.0) else 0.0)
F1 += femm_left.distrib_loads(geom1, T1, fi_1, 3)

########################################################################################################################
# Side right
########################################################################################################################

if right_m == "quad":
    xs2 = np.linspace(1.0, 2.0, int(N_elem2/2)+1)
    ys2 = np.linspace(0.0, 2.0, N_elem2+1)
    fens2, fes2 = q4_blockx(xs2, ys2)
    femm2 = FEMMHeatDiff(fes = fes2, material=m, integration_rule=GaussRule(dim=2, order=2))

else:
    fens2, fes2 = t3_ablock(1, 2, int(N_elem2/2), N_elem2)
    fens2.xyz[:, 0] += 1.0
    femm2 = FEMMHeatDiff(fes = fes2, material=m, integration_rule=TriRule(npts=1))

T2 = NodalField(nfens=fens2.count(), dim=1)
geom2 = NodalField(fens=fens2)
dbc_box2 = bounding_box([2.0, 0.0])
dbc_nodes2 = fenode_select(fens2, dbc_box2)
T2.set_ebc(dbc_nodes2, val=1.0)
T2.apply_ebc()
T2.numberdofs()
K2 = femm2.conductivity(geom2, T2)
F2=femm2.nz_ebc_loads_conductivity(geom2, T2)
boundary_fes2 = mesh_boundary(fes2)
femm_right = FEMMHeatDiff(fes = boundary_fes2, material=m, integration_rule=GaussRule(dim=1, order=2))
fi_2 = ForceIntensity(magn=lambda x, J: 1.0 if np.isclose(x[0], 2.0) else 0.0)
F2 += femm_right.distrib_loads(geom2, T2, fi_2, 3)

########################################################################################################################
# Interface
########################################################################################################################
# ys_i = np.unique(np.hstack([fens2.xyz[:, 1],fens1.xyz[:, 1]]))
ys_i = np.linspace(0.0, 2.0, N_elem_i+1)  # x-coordinates
xs_i = np.full_like(ys_i, 1.0)     # y-coordinates (constant)
fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)

mu =  ElementalField(nelems=fes_i.count(), dim=1)
geom_i = NodalField(fens=fens_i)
mu.numberdofs()
femm_i = FEMMHeatDiff(fes = fes_i, material=m, integration_rule=GaussRule(dim=1, order=1))
M = femm_i.lam_mat(geom_i, mu)

########################################################################################################################
# Mapping
########################################################################################################################
box = bounding_box(fens_i.xyz)
boundary_nodes1 = fenode_select(fens1, box)
boundary_nodes2 = fenode_select(fens2, box)

interface_fe_idx1 = fe_select(fens1, boundary_fes1, box=box)
interface_fe_idx2 = fe_select(fens2, boundary_fes2, box=box)

# operator mapping from subdomain boundary to interface
g1 = assemble_gamma(fens1, boundary_fes1, interface_fe_idx1, fens_i)
g2 = assemble_gamma(fens2, boundary_fes2, interface_fe_idx2, fens_i)


# bottom row blocks
###################
B1 = M@g1
B2 = -M@g2
# remove dbc_nodes columns
# B1 = np.delete(B1, dbc_nodes1, axis=1)
B2 = np.delete(B2, dbc_nodes2, axis=1)

# right column blocks
###################

G1 = build_edge_map_simple(fens_i.xyz, fens1.xyz, boundary_fes1.conn, interface_fe_idx1)
G2 = -build_edge_map_simple(fens_i.xyz, fens2.xyz, boundary_fes2.conn, interface_fe_idx2 )

# G1 = np.delete(G1.toarray(), dbc_nodes1, axis=0)
G2 = np.delete(G2.toarray(), dbc_nodes2, axis=0)


A = bmat([
    [K1,    None,   G1],
    [None,  K2,     G2],
    [B1,    B2,     None],
], format='csr')

print(f"Dim - {A.shape}\n Rank - {np.linalg.matrix_rank(A.toarray())}")

F = np.concatenate([F1, F2, np.zeros(fes_i.count())])
U = spsolve(A, F)

########################################################################################################################
# Post Processing
########################################################################################################################
# Output files
T1.scatter_sysvec(U[0:K1.shape[0]])
T2.scatter_sysvec(U[K1.shape[0]:K1.shape[0]+K2.shape[0]])

script_path = __file__
script_filename = os.path.basename(script_path)[:-3]
if not os.path.exists(script_filename):
    os.mkdir(script_filename)
subdir = f"{left_m}-{right_m}-{N_elem1}-{N_elem_i}-{N_elem2}"
script_filename = os.path.join(script_filename, subdir)
if not os.path.exists(script_filename):
    os.mkdir(script_filename)


exact =  lambda x: x[0]-1
L2_err1 = L2_err(femm1, geom1, T1, exact)
L2_err2 = L2_err(femm2, geom2, T2, exact)
vtkexport(f"{script_filename}/left", fes1, geom1, {"temp":T1, "err":L2_err1})
vtkexport(f"{script_filename}/right", fes2, geom2, {"temp":T2, "err":L2_err2})
vtkexport(f"{script_filename}/left", fes1, geom1, {"temp":T1, "err":L2_err1})
vtkexport(f"{script_filename}/right", fes2, geom2, {"temp":T2, "err":L2_err2})
merge_vtk_files_common_fields(f"{script_filename}/left.vtu", f"{script_filename}/right.vtu", f"{script_filename}/merged.vtu")
print(f"Maximum L2 error on left = {np.max(L2_err1.values)} \n"
      f"Maximum L2 error on right = {np.max(L2_err2.values)}")

# plotting lagrange multiplier
mu.scatter_sysvec(U[K1.shape[0]+K2.shape[0]:])
print(f"Lambda values : {mu.values.T}")
print(f"sum of lambda values = {np.sum(mu.values)}")
import matplotlib.pyplot as plt
plt.stairs((U[K1.shape[0]+K2.shape[0]:]),fens_i.xyz[:,1], baseline=None,  label="lambda f")
plt.legend()
plt.title("Lagrange multipliers and their projections\n NBC on top and bottom")
plt.xlabel("y along the interface")
plt.ylabel("Lagrange multiplier")
# # plt.ylim(-50,50)
plt.savefig(f"{script_filename}/lagrange.png")
plt.show()

# plotting solution along the interface:
box_ = np.array([1,1,0,2])
x = np.linspace(0.0, 2.0, 100)
x0 = np.zeros_like(x)
x05 = np.zeros_like(x)+0.5
x1 = np.zeros_like(x)+1.0
y = 0*x
plt.plot(fens1.xyz[fenode_select(fens1, box_),1], T1.values[fenode_select(fens1, box_)], "-x", label="T1")
plt.plot(fens2.xyz[fenode_select(fens2, box_),1], T2.values[fenode_select(fens2, box_)], "o", linestyle='--',label="T2")
plt.plot(x,y, label="exact")
plt.title("solution along the interface:\n uniform node distribution on frame")
plt.xlabel("y along the interface")
plt.ylabel("Temperature")
plt.ylim(-1,1)
plt.legend()
plt.savefig(f"{script_filename}/interface_sol.png")
plt.show()

# integral of difference
t1_b = T1.values[fenode_select(fens1, box_)].flatten()
y1_b = fens1.xyz[fenode_select(fens1, box_), 1]
t2_b = T2.values[fenode_select(fens2, box_)].flatten()
y2_b = fens2.xyz[fenode_select(fens2, box_), 1]
t2_b_on1 = np.interp(y1_b, y2_b, t2_b)
t_integ =trapezoid(np.abs(t1_b-t2_b_on1), y1_b)
print(f"∫(T1-T2)dΓ = {t_integ}")

########################################################################################################################
# pyvista
########################################################################################################################
use_pv = True
if use_pv:
    merged_path = os.path.join(script_filename, "merged.vtu")
    mesh = pv.read(merged_path)
    pv.global_theme.show_scalar_bar = True
    pv.global_theme.axes.show = False
    pv.global_theme.font.label_size = 14
    pv.global_theme.font.title_size = 16
    pv.global_theme.colorbar_orientation = 'vertical'

    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)
    plotter.subplot(0, 0)

    plotter.add_text(
        "Temperature",
        font_size=18,
        position="upper_edge",  # centered top
        color="black",
    )
    plotter.add_mesh(
        mesh,
        scalars="temp",
        # preference=_pref(mesh, "temp"),
        cmap = "coolwarm",
        show_edges=True,
    )
    plotter.camera_position = "xy"  # top-down for 2D meshes
    plotter.subplot(0, 1)

    plotter.add_text(
        "Error (L2)",
        font_size=18,
        position="upper_edge",  # centered top
        color="black",
    )
    plotter.add_mesh(
        mesh,
        scalars="err",
        # preference=_pref(mesh, "err"),
        cmap="coolwarm",
        show_edges=True,
    )
    plotter.camera_position = "xy"
    plotter.link_views()
    out_png = os.path.join(script_filename, "temp_err.png")
    # plotter.screenshot(out_png)
    plotter.show(screenshot=out_png)
    plotter.close()