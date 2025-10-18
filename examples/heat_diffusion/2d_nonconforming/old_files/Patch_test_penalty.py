import os
import sys

from scipy.special import lmbda

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
from utilities import assemble_gamma
k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]), rho=1.0)



xs1 = np.linspace(0.0, 1.0, 8)
ys1 = np.linspace(0.0, 2.0, 12)
fens1, fes1 = q4_blockx(xs1, ys1)
femm1 = FEMMHeatDiff(fes = fes1, material=m, integration_rule=GaussRule(dim=2, order=2))
T1 = NodalField(nfens=fens1.count(), dim=1)
geom1 = NodalField(fens=fens1)
T1.numberdofs()
K1 = femm1.conductivity(geom1, T1)
F1=femm1.nz_ebc_loads_conductivity(geom1, T1)
boundary_fes1 = mesh_boundary(fes1)

femm_left = FEMMHeatDiff(fes = boundary_fes1, material=m, integration_rule=GaussRule(dim=1, order=2))
fi_1 = ForceIntensity(magn=lambda x, J: -1.0 if np.isclose(x[0], 0.0) else 0.0)
F1 += femm_left.distrib_loads(geom1, T1, fi_1, 3)
dbc_nodes1 = []


xs2 = np.linspace(1.0, 2.0, 10)
ys2 = np.linspace(0.0, 2.0, 22)
fens2, fes2 = q4_blockx(xs2, ys2)
femm2 = FEMMHeatDiff(fes = fes2, material=m, integration_rule=GaussRule(dim=2, order=2))
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

N=32
ys_i = np.linspace(0.0, 2.0, N)  # x-coordinates
xs_i = np.full_like(ys_i, 1.0)     # y-coordinates (constant)
fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)

mu =  NodalField(nfens=fens_i.count(), dim=1)
geom_i = NodalField(fens=fens_i)
mu.numberdofs()
femm_i = FEMMHeatDiff(fes = fes_i, material=m, integration_rule=GaussRule(dim=1, order=2))
M = femm_i.mass(geom_i, mu)

box = bounding_box(fens_i.xyz)
boundary_nodes1 = fenode_select(fens1, box)
boundary_nodes2 = fenode_select(fens2, box)



interface_fe_idx1 = fe_select(fens1, boundary_fes1, box=box)
interface_fe_idx2 = fe_select(fens2, boundary_fes2, box=box)


g1 = assemble_gamma(fens1, boundary_fes1, interface_fe_idx1, fens_i)
g2 = assemble_gamma(fens2, boundary_fes2, interface_fe_idx2, fens_i)

beta = 1e8
B11 = beta*g1.T@M@g1
B22 = beta*g2.T@M@g2
B12 = beta*g1.T@M@g2
B21 = beta*g2.T@M@g1
# remove dbc_nodes columns
# B1 = np.delete(B1, dbc_nodes1, axis=1)
# B2 = np.delete(B2, dbc_nodes2, axis=1)

B11 = np.delete(B11, dbc_nodes1, axis=1)
B11 = np.delete(B11, dbc_nodes1, axis=0)

B22 = np.delete(B22, dbc_nodes2, axis=0)
B22 = np.delete(B22, dbc_nodes2, axis=1)

B12 = np.delete(B12, dbc_nodes1, axis=0)
B12 = np.delete(B12, dbc_nodes2, axis=1)

B21 = np.delete(B21, dbc_nodes2, axis=0)
B21 = np.delete(B21, dbc_nodes1, axis=1)

# mat_size = K1.shape[0] + K2.shape[0] + B1.shape[0] + B2.shape[0]


A = bmat([
    [K1+ B11,   -B12],
    [-B21,  K2 + B22,],
], format='csr')


F = np.concatenate([F1, F2])
U = spsolve(A, F)

T1.scatter_sysvec(U[0:K1.shape[0]])
T2.scatter_sysvec(U[K1.shape[0]:K1.shape[0]+K2.shape[0]])

import os
script_path = __file__
script_filename = os.path.basename(script_path)[:-3]
if not os.path.exists(script_filename):
    os.mkdir(script_filename)

from utilities import L2_err
exact =  lambda x: x[0]-1
L2_err1 = L2_err(femm1, geom1, T1, exact)
L2_err2 = L2_err(femm2, geom2, T2, exact)

vtkexport(f"{script_filename}/left", fes1, geom1, {"temp":T1, "err":L2_err1})
vtkexport(f"{script_filename}/right", fes2, geom2, {"temp":T2, "err":L2_err2})