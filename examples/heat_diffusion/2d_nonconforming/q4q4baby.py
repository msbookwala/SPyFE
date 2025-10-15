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
from spyfe.integ_rules import GaussRule
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
from utilities import assemble_gamma, L2_err




# N_i = 25
# ys_i = np.linspace(0.0, 1.0, N_i)  # x-coordinates
# xs_i = np.full_like(ys_i, 0.5)     # y-coordinates (constant)
# fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)
box = np.array([0.5,0.5,0.0,1.0])

box_ = np.array([0.5,0.5,0.0,1.0])
box[2]+=1e-5
box[3]-=1e-5


start0 = time.time()

# These are the constants in the problem, k is kappa
boundaryf = lambda x, y: 1.0 + x ** 2 + 2 * y ** 2
Q = -6  # internal heat generation rate
k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]), rho=1.0)
Dz = 1.0  # thickness of the slice

########################################################################################################################
# subdomain 1
########################################################################################################################
N1 = 40
xs1 = np.linspace(0.0, 0.5, 2)
ys1 = np.linspace(0.0, 1.0, 4)
fens1, fes1 = q4_blockx(xs1, ys1)
bfes1 = mesh_boundary(fes1)
boundary_nodes1 = fenode_select(fens1, box)
boundary_fes1 = mesh_boundary(fes1)

cn1 = connected_nodes(bfes1)
geom1 = NodalField(fens=fens1)
T1 = NodalField(nfens=fens1.count(), dim=1)

dbc_nodes1 = np.setdiff1d(cn1, boundary_nodes1, assume_unique=True)
for index  in dbc_nodes1:
    T1.set_ebc([index], val=boundaryf(fens1.xyz[index, 0], fens1.xyz[index, 1]))
T1.apply_ebc()
femm1 = FEMMHeatDiff(material=m, fes=fes1, integration_rule=GaussRule(dim=2, order=2))

T1.numberdofs()
fi1= ForceIntensity(magn=lambda x, J: Q)
F1 = femm1.distrib_loads(geom1, T1, fi1, 3)
F1 += femm1.nz_ebc_loads_conductivity(geom1, T1)
K1 = femm1.conductivity(geom1, T1)
interface_fe_idx1 = fe_select(fens1, boundary_fes1, box=box_)


########################################################################################################################
# subdomain 2
########################################################################################################################
N2 = 30
xs2 = np.linspace(0.5, 1.0, 2)
ys2 = np.linspace(0.0, 1.0, 4)
fens2, fes2 = q4_blockx(xs2, ys2)
bfes2 = mesh_boundary(fes2)
boundary_nodes2 = fenode_select(fens2, box)
boundary_fes2 = mesh_boundary(fes2)
cn2 = connected_nodes(bfes2)
geom2 = NodalField(fens=fens2)
T2 = NodalField(nfens=fens2.count(), dim=1)
dbc_nodes2 = np.setdiff1d(cn2, boundary_nodes2, assume_unique=True)
for index  in dbc_nodes2:
    T2.set_ebc([index], val=boundaryf(fens2.xyz[index, 0], fens2.xyz[index, 1]))
T2.apply_ebc()
femm2 = FEMMHeatDiff(material=m, fes=fes2, integration_rule=GaussRule(dim=2, order=2))

T2.numberdofs()
fi2 = ForceIntensity(magn=lambda x, J: Q)
F2 = femm2.distrib_loads(geom2, T2, fi2, 3)
F2 += femm2.nz_ebc_loads_conductivity(geom2, T2)
K2 = femm2.conductivity(geom2, T2)
interface_fe_idx2 = fe_select(fens2, boundary_fes2, box=box_)
########################################################################################################################
# interface
########################################################################################################################
# N_i = 40
# ys_i = np.linspace(0.0, 1.0, N_i+1)  # x-coordinates
# xs_i = np.full_like(ys_i, 0.5)     # y-coordinates (constant)
# fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)


# # nodes to create frame including the ones that go for dbc

bn1 = fenode_select(fens1, box_)
bn2 = fenode_select(fens2, box_)
xys = np.unique(np.round(np.vstack([fens1.xyz[bn1], fens2.xyz[bn2]]), 7), axis=0)
ys_i = xys[:,1]
xs_i = xys[:,0]
fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)

mu =  NodalField(nfens=fens_i.count(), dim=1)
geom_i = NodalField(fens=fens_i)
mu.numberdofs()
femm_i = FEMMHeatDiff(fes = fes_i, material=m, integration_rule=GaussRule(dim=1, order=2))
M = femm_i.mass(geom_i, mu)

g1 = assemble_gamma(fens1, boundary_fes1, interface_fe_idx1, fens_i)
g2 = assemble_gamma(fens2, boundary_fes2, interface_fe_idx2, fens_i)
B1 = M@g1
B2 = -M@g2

B1_p  = B1[1:-1, dbc_nodes1]
B2_p  = B2[1:-1, dbc_nodes2]
T1_p = T1.fixed_values[T1.is_fixed]
T2_p = T2.fixed_values[T2.is_fixed]

dbc_lam_f = -B1_p@T1_p - B2_p@T2_p

# remove dbc_nodes columns
B1 = np.delete(B1, dbc_nodes1, axis=1)
B2 = np.delete(B2, dbc_nodes2, axis=1)

B1 = B1[1:-1,:]
B2 = B2[1:-1,:]



A = bmat([
    [K1,    None,   B1.T],
    [None,  K2,     B2.T],
    [B1,    B2,     None],
], format='csr')


# F = np.concatenate([F1, F2, np.zeros(fens_i.count())])
F = np.concatenate([F1, F2, dbc_lam_f])
U = spsolve(A, F)
T1.scatter_sysvec(U[0:K1.shape[0]])
T2.scatter_sysvec(U[K1.shape[0]:K1.shape[0]+K2.shape[0]])


import os
script_path = __file__
script_filename = os.path.basename(script_path)[:-3]
if not os.path.exists(script_filename):
    os.mkdir(script_filename)
# vtkexport(f"{script_filename}/left", fes1, geom1, {"temp":T1})
# vtkexport(f"{script_filename}/right", fes2, geom2, {"temp":T2})

exact =  lambda x: 1.0 + x[0] ** 2 + 2 * x[1] ** 2
L2_err1 = L2_err(femm1, geom1, T1, exact)
L2_err2 = L2_err(femm2, geom2, T2, exact)

vtkexport(f"{script_filename}/left", fes1, geom1, {"temp":T1, "err":L2_err1})
vtkexport(f"{script_filename}/right", fes2, geom2, {"temp":T2, "err":L2_err2})
from mergevtk import merge_vtk_files_common_fields
merge_vtk_files_common_fields(f"{script_filename}/left.vtu", f"{script_filename}/right.vtu", f"{script_filename}/merged.vtu")

# mu.scatter_sysvec(U[K1.shape[0]+K2.shape[0]:])
g1_plus = np.linalg.pinv(g1)[boundary_nodes1, 1:-1]
g2_plus = np.linalg.pinv(g2)[boundary_nodes2, 1:-1]
lmbd_f = U[K1.shape[0]+K2.shape[0]:]
print(f"Lambda values : {U[K1.shape[0]+K2.shape[0]:]}")
print(f"sum of lambda values = {np.sum(U[K1.shape[0]+K2.shape[0]:])/len(U[K1.shape[0]+K2.shape[0]:])}")

lmbd1 = g1_plus@lmbd_f
lmbd2 = g2_plus@lmbd_f

import matplotlib.pyplot as plt
plt.plot(fens_i.xyz[1:-1, 1],(U[K1.shape[0]+K2.shape[0]:]), label="lambda f")
plt.plot(fens1.xyz[boundary_nodes1, 1],lmbd1, label="lambda 1")
plt.plot(fens2.xyz[boundary_nodes2, 1], lmbd2, label="lambda 2")
plt.legend()
plt.show()

