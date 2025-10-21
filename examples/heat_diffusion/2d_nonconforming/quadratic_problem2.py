import os
import sys
from scipy.special import lmbda
from spyfe.meshing.boxes import bounding_box
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.quadrilaterals import q4_blockx
from spyfe.meshing.generators.triangles import t3_ablock
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
import numpy as np
from numpy import array
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.femms.femm_defor import FEMMDefor
from spyfe.fields.nodal_field import NodalField
from spyfe.fields.elemental_field import ElementalField
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
from scipy.integrate import trapezoid

N_elem1 = 30
N_elem2 = 29
N_elem_i = min(N_elem1, N_elem2)
# N_elem_i = 20
left_m = "q"
right_m = "t"
skew = 0.0
top_bc = "D"
elem_lagrange= True

# These are the constants in the problem, k is kappa
boundaryf = lambda x, y: 1.0 + x ** 2 + 2 * y ** 2
Q = -6  # internal heat generation rate
k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]), rho=1.0)
Dz = 1.0  # thickness of the slice
q = lambda x, y: 4*y

box = np.array([0.5,0.5,0.0,1.0])
box_ = np.array([0.5,0.5,0.0,1.0])
box_left = np.array([0.0,0.0,0.0,1.0])
box_right = np.array([1.0,1.0,0.0,1.0])
box_top = np.array([0.0,1.0,1.0,1.0])
box_bottom = np.array([1.0,0.0,0.0,0.0])
########################################################################################################################
# subdomain 1
########################################################################################################################
if left_m == "q":
    xs1 = np.linspace(0.0, 0.5, int(N_elem1/2)+1)
    ys1 = np.linspace(0.0, 1.0, N_elem1+1)



    fens1, fes1 = q4_blockx(xs1, ys1)
else:
    fens1, fes1 = t3_ablock(1, 2, int(N_elem1/2), N_elem1)

# extracting the interface information before skewing the mesh
boundary_fes1 = mesh_boundary(fes1)
iedge_nodes1 = fenode_select(fens1, box)
interface_fe_idx1 = fe_select(fens1, boundary_fes1, box=box)
fens1.xyz[:, 0] +=  fens1.xyz[:, 0]*(fens1.xyz[:, 1]-0.5) * skew

cn1 = connected_nodes(boundary_fes1)
geom1 = NodalField(fens=fens1)
T1 = NodalField(nfens=fens1.count(), dim=1)

if top_bc=="D":
    # bt = [0, 0.5 - 1e-6, 1, 1]
    # bb = [0, 0.5 - 1e-6, 0, 0]
    dbc_nodes1 = np.sort(np.unique(np.hstack([
                                     fenode_select(fens1, box_left),
                                     fenode_select(fens1, box_right),
                                     fenode_select(fens1, box_top),
                                     fenode_select(fens1, box_bottom)
                                     ])))
else:
    dbc_nodes1 = fenode_select(fens1, box_left)
dbc_nodes1 = dbc_nodes1.astype(np.int32)
for index  in dbc_nodes1:
    T1.set_ebc([index], val=boundaryf(fens1.xyz[index, 0], fens1.xyz[index, 1]))
T1.apply_ebc()

if left_m == "q":
    femm1 = FEMMHeatDiff(fes = fes1, material=m, integration_rule=GaussRule(dim=2, order=2))
else:
    femm1 = FEMMHeatDiff(fes = fes1, material=m, integration_rule=TriRule(npts=1))

T1.numberdofs()
fi1= ForceIntensity(magn=lambda x, J: Q)
F1 = femm1.distrib_loads(geom1, T1, fi1, 3)
F1 += femm1.nz_ebc_loads_conductivity(geom1, T1)
K1 = femm1.conductivity(geom1, T1)

fi_bottom = ForceIntensity(magn=lambda x, J: q(x[0], x[1]) if np.isclose(x[1], 0.0) else 0.0)
fi_top = ForceIntensity(magn=lambda x, J: q(x[0], x[1]) if np.isclose(x[1], 1.0) else 0.0)
if top_bc=="N":
    femm_nbc1 = FEMMHeatDiff(fes = boundary_fes1, material=m, integration_rule=GaussRule(dim=1, order=2))
    F1 += femm_nbc1.distrib_loads(geom1, T1, fi_bottom, 3)
    F1 += femm_nbc1.distrib_loads(geom1, T1, fi_top, 3)

########################################################################################################################
# subdomain 2
########################################################################################################################
if right_m == "q":
    xs2 = np.linspace(0.5, 1, int(N_elem2/2)+1)
    ys2 = np.linspace(0.0, 1.0, N_elem2+1)
    fens2, fes2 = q4_blockx(xs2, ys2)
else:
    fens2, fes2 = t3_ablock(0.5, 1, int(N_elem2/2), N_elem2)
    fens2.xyz[:, 0] += 0.5

# extracting the interface information before skewing the mesh
boundary_fes2 = mesh_boundary(fes2)
iedge_nodes2 = fenode_select(fens2, box)
interface_fe_idx2 = fe_select(fens2, boundary_fes2, box=box)
fens2.xyz[:, 0] +=  (1-fens2.xyz[:, 0])*(fens2.xyz[:, 1] - 0.5) * skew

cn2 = connected_nodes(boundary_fes2)
geom2 = NodalField(fens=fens2)
T2 = NodalField(nfens=fens2.count(), dim=1)
if top_bc=="D":
    # box_top = [0.5+1e-6,1,1,1]
    # box_bottom = [0.5+1e-6,1,0,0]
    dbc_nodes2 = np.sort(np.unique(np.hstack([
                                     fenode_select(fens2, box_left),
                                     fenode_select(fens2, box_right),
                                     fenode_select(fens2, box_top),
                                     fenode_select(fens2, box_bottom)
                                     ])))
else:
    dbc_nodes2 = fenode_select(fens2, box_right)
dbc_nodes2 = dbc_nodes2.astype(np.int32)

for index  in dbc_nodes2:
    T2.set_ebc([index], val=boundaryf(fens2.xyz[index, 0], fens2.xyz[index, 1]))
T2.apply_ebc()

if right_m == "q":
    femm2 = FEMMHeatDiff(fes = fes2, material=m, integration_rule=GaussRule(dim=2, order=2))
else:
    femm2 = FEMMHeatDiff(fes = fes2, material=m, integration_rule=TriRule(npts=1))

T2.numberdofs()
fi2 = ForceIntensity(magn=lambda x, J: Q)
F2 = femm2.distrib_loads(geom2, T2, fi2, 3)
F2 += femm2.nz_ebc_loads_conductivity(geom2, T2)
K2 = femm2.conductivity(geom2, T2)

if top_bc=="N":
    femm_nbc2 = FEMMHeatDiff(fes = boundary_fes2, material=m, integration_rule=GaussRule(dim=1, order=2))
    F2 += femm_nbc2.distrib_loads(geom2, T2, fi_bottom, 3)
    F2 += femm_nbc2.distrib_loads(geom2, T2, fi_top, 3)

########################################################################################################################
# interface
########################################################################################################################
ys_i = np.linspace(0.0, 1.0, N_elem_i+1)  # y-coordinates
xs_i = np.full_like(ys_i, 0.5)     # x-coordinates (constant)

# xys = np.unique(np.round(np.vstack([fens1.xyz[iedge_nodes1], fens2.xyz[iedge_nodes2]]), 7), axis=0)
# ys_i = xys[:,1]
# xs_i = xys[:,0]

fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)
fens_i.xyz[:, 0] +=  fens_i.xyz[:, 0]*(fens_i.xyz[:, 1]-0.5) * skew

if elem_lagrange:
    mu =  ElementalField(nelems=fes_i.count(), dim=1)
    n_lambda = fes_i.count()
else:
    mu =  NodalField(nfens=fens_i.count(), dim=1)
    n_lambda = fens_i.count()

geom_i = NodalField(fens=fens_i)
mu.numberdofs()

femm_i = FEMMHeatDiff(fes = fes_i, material=m, integration_rule=GaussRule(dim=1, order=2))
if elem_lagrange:
    M = femm_i.lam_mat(geom_i, mu)
else:
    M = femm_i.mass(geom_i, mu)

########################################################################################################################
# Mapping
########################################################################################################################

# ---- Build consistent L2 cross-mass on the interface (right side shown) ----
# Frame nodes (already ordered by construction, but ok to recompute s):
frame_xyz = fens_i.xyz

# Right: order the *subdomain* interface nodes via connectivity
edge_conn2 = boundary_fes2.conn[interface_fe_idx2]
C2_edge, edge_nodes2_ordered = cross_mass_P1_frame_P1_sub(
    frame_xyz=frame_xyz,
    sub_xyz=fens2.xyz,
    sub_edge_conn=edge_conn2
)
# Embed to full right-side node space:
C2 = embed_cross_mass_to_full(C2_edge, edge_nodes2_ordered, fens2.count())

# Left side similarly:
edge_conn1 = boundary_fes1.conn[interface_fe_idx1]
C1_edge, edge_nodes1_ordered = cross_mass_P1_frame_P1_sub(
    frame_xyz=frame_xyz,
    sub_xyz=fens1.xyz,
    sub_edge_conn=edge_conn1
)
C1 = embed_cross_mass_to_full(C1_edge, edge_nodes1_ordered, fens1.count())

# Bottom row (constraints) and right column (adjoint!) — *no other maps*:
B1 =  C1.copy()
B2 = -C2.copy()
# Dirichlet handling (as you do now)

B1_p = B1[:, dbc_nodes1]
B2_p = B2[:, dbc_nodes2]
T1_p = T1.fixed_values[T1.is_fixed]
T2_p = T2.fixed_values[T2.is_fixed]
dbc_lam_f = -(B1_p @ T1_p) - (B2_p @ T2_p)
B1 = csr_matrix(np.delete(B1.toarray(), dbc_nodes1, axis=1))
B2 = csr_matrix(np.delete(B2.toarray(), dbc_nodes2, axis=1))

C1 = build_p0p1_interpolator_frame_to_sub_full__no_reuse(frame_xyz, fens1.xyz, boundary_fes1.conn, interface_fe_idx1)
C2 = -build_p0p1_interpolator_frame_to_sub_full__no_reuse(frame_xyz, fens2.xyz, boundary_fes2.conn, interface_fe_idx2)
C1_p = C1[:, dbc_nodes1]
C2_p = C2[:, dbc_nodes2]
T1_p = T1.fixed_values[T1.is_fixed]
T2_p = T2.fixed_values[T2.is_fixed]
dbc_lam_f = -(C1_p @ T1_p) - (C2_p @ T2_p)
C2 = np.delete(C2.toarray(), dbc_nodes2, axis = 1)
C1 = np.delete(C1.toarray(), dbc_nodes1, axis = 1)

A = bmat([
    [K1,    None,   C1.T],
    [None,  K2,     C2.T],
    [C1,    C2,     None],
], format='csr')

# A = bmat([
#     [K1,    None,   B1.T],
#     [None,  K2,     B2.T],
#     [B1,    B2,     None],
# ], format='csr')
# A = bmat([
#     [K1,    None,   G1],
#     [None,  K2,     G2],
#     [B1,    B2,     None],
# ], format='csr')
print(f"Dim - {A.shape}\n Rank - {np.linalg.matrix_rank(A.toarray())}")
F = np.concatenate([F1, F2, dbc_lam_f])
U = spsolve(A, F)
########################################################################################################################
# Post Processing
########################################################################################################################
# Output files
T1.scatter_sysvec(U[0:K1.shape[0]])
T2.scatter_sysvec(U[K1.shape[0]:K1.shape[0]+K2.shape[0]])
mu.scatter_sysvec(U[K1.shape[0]+K2.shape[0]:])


script_path = __file__
script_filename = os.path.basename(script_path)[:-3]
if not os.path.exists(script_filename):
    os.mkdir(script_filename)
subdir = f"{left_m}-{right_m}-{N_elem1}-{N_elem_i}-{N_elem2}-skew-{int(100*skew)}-{top_bc}"
script_filename = os.path.join(script_filename, subdir)
if not os.path.exists(script_filename):
    os.mkdir(script_filename)

exact =  lambda x: 1.0 + np.pow(x[0],2 )+ 2 * np.pow(x[1], 2)
L2_err1 = L2_err(femm1, geom1, T1, exact)
L2_err2 = L2_err(femm2, geom2, T2, exact)
vtkexport(f"{script_filename}/left", fes1, geom1, {"Temperature":T1, "Error":L2_err1})
vtkexport(f"{script_filename}/right", fes2, geom2, {"Temperature":T2, "Error":L2_err2})
merge_vtk_files_common_fields(f"{script_filename}/left.vtu", f"{script_filename}/right.vtu", f"{script_filename}/merged.vtu")
print(f"Maximum L2 error on left = {np.max(L2_err1.values)} \n"
      f"Maximum L2 error on right = {np.max(L2_err2.values)}")


# plotting lagrange multiplier

print(f"Lambda values : {mu.values.T}")
print(f"sum of lambda values = {np.sum(mu.values)}")
import matplotlib.pyplot as plt
if elem_lagrange:
    plt.stairs(mu.values.flatten(), fens_i.xyz[:,1], baseline=None,  label="lambda f")
else:
    plt.plot(fens_i.xyz[:,1], mu.values.flatten(),   label="lambda f")
plt.legend()
plt.title("Lagrange multipliers and their projections\n NBC on top and bottom")
plt.xlabel("y along the interface")
plt.ylabel("Lagrange multiplier")
# # plt.ylim(-50,50)
plt.savefig(f"{script_filename}/lagrange.png")
plt.show()

# plotting solution along the interface:
freq = 100
xyz = np.zeros((freq,2))
xyz[:,0] = np.linspace((fens1.xyz[iedge_nodes1[0],0]), (fens1.xyz[iedge_nodes1[-1],0]), freq)
xyz[:,1] = np.linspace((fens1.xyz[iedge_nodes1[0],1]), (fens1.xyz[iedge_nodes1[-1],1]), freq)
dist = np.linalg.norm(xyz - xyz[0,:], axis=1)
y = np.array([exact(xyz[i,:]) for i in range(freq)])
plt.plot(np.linalg.norm(fens1.xyz[iedge_nodes1,:] - fens1.xyz[iedge_nodes1[0],:], axis=1), T1.values[iedge_nodes1], "-x", label="T1")
plt.plot(np.linalg.norm(fens2.xyz[iedge_nodes2,:] - fens2.xyz[iedge_nodes2[0],:], axis=1), T2.values[iedge_nodes2], "o", linestyle='--', label="T2")
plt.plot(dist, y, label="exact")
plt.title("solution along the interface:\n uniform node distribution on frame")
plt.xlabel("y along the interface")
plt.ylabel("Temperature")
# plt.ylim(-1,1)
plt.legend()
plt.savefig(f"{script_filename}/interface_sol.png")
plt.show()

# integral of difference
t1_b = T1.values[iedge_nodes1].flatten()
y1_b = fens1.xyz[iedge_nodes1, 1]
t2_b = T2.values[iedge_nodes2].flatten()
y2_b = fens2.xyz[iedge_nodes2, 1]
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
    pv.global_theme.colorbar_orientation = 'horizontal'
    pv.global_theme.colorbar_horizontal.position_x = 0.2

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
        scalars="Temperature",
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
        scalars="Error",
        # preference=_pref(mesh, "err"),
        cmap="coolwarm",
        show_edges=True,
    )
    plotter.camera_position = "xy"
    plotter.link_views()
    out_png = os.path.join(script_filename, "temp_err.png")
    # plotter.screenshot(out_png)
    # plotter.image_scale = 4
    plotter.show(screenshot=out_png)
    plotter.close()




