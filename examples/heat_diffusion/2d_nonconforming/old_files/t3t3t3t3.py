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
from utilities import assemble_gamma, L2_err



N_i1 = 10
ys_i1 = np.linspace(0.0, 0.5, N_i1)  # x-coordinates
xs_i1 = np.full_like(ys_i1, 0.5)     # y-coordinates (constant)
fens_i1, fes_i1 = l2_blockx_2D(xs_i1, ys_i1)
box1 = bounding_box(fens_i1.xyz)
box1[2]+=1e-5


N_i2 = 10
xs_i2 = np.linspace(0.5, 1.0, N_i2)
ys_i2 = np.full_like(xs_i2, 0.5)
fens_i2, fes_i2 = l2_blockx_2D(xs_i2, ys_i2)
box2 = bounding_box(fens_i2.xyz)
box2[1]-=1e-5

N_i3 = 20
ys_i3 = np.linspace(0.5, 1.0, N_i3)
xs_i3 = np.full_like(ys_i3, 0.5)
fens_i3, fes_i3 = l2_blockx_2D(xs_i3, ys_i3)
box3 = bounding_box(fens_i3.xyz)
box3[3]-=1e-5

N_i4 = 5
xs_i4 = np.linspace(0.0, 0.5, N_i4)
ys_i4 = np.full_like(xs_i4, 0.5)
fens_i4, fes_i4 = l2_blockx_2D(xs_i4, ys_i4)
box4 = bounding_box(fens_i4.xyz)
box4[0]+=1e-5

boxes = [box1, box2, box3, box4]



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
N1 = 5
# xs1 = np.linspace(0.0, 0.5, int(N1 / 2) + 1)
# ys1 = np.linspace(0.0, 1.0, N1 + 1)
# fens1, fes1 = q4_blockx(xs1, ys1)
fens1, fes1 = t3_ablock(0.5, 0.5, N1, N1)
bfes1 = mesh_boundary(fes1)
boundary_nodes1 = np.unique(np.hstack([fenode_select(fens1, box) for box in boxes ]))
boundary_fes1 = mesh_boundary(fes1)

cn1 = connected_nodes(bfes1)
geom1 = NodalField(fens=fens1)
T1 = NodalField(nfens=fens1.count(), dim=1)

dbc_nodes1 = np.setdiff1d(cn1, boundary_nodes1, assume_unique=True)
for index  in dbc_nodes1:
    T1.set_ebc([index], val=boundaryf(fens1.xyz[index, 0], fens1.xyz[index, 1]))
T1.apply_ebc()
femm1 = FEMMHeatDiff(material=m, fes=fes1, integration_rule=TriRule(npts=3))

T1.numberdofs()
fi1= ForceIntensity(magn=lambda x, J: Q)
F1 = femm1.distrib_loads(geom1, T1, fi1, 3)
F1 += femm1.nz_ebc_loads_conductivity(geom1, T1)
K1 = femm1.conductivity(geom1, T1)
interface_fe_idx1 = np.unique(np.hstack([fe_select(fens1, boundary_fes1, box=box) for box in boxes]))

err1 = ElementalField(fes = fes1)

########################################################################################################################
# subdomain 2
########################################################################################################################
N2 = 10
# xs2 = np.linspace(0.5, 1.0, int(N2 / 2) + 1)
# ys2 = np.linspace(0.0, 1.0, N2 + 1)
# fens2, fes2 = q4_blockx(xs2, ys2)
fens2,fes2 = t3_ablock(0.5, 0.5, N2, N2)
fens2.xyz[:, 0] += 0.5

bfes2 = mesh_boundary(fes2)
boundary_nodes2 = np.unique(np.hstack([fenode_select(fens2, box) for box in boxes ]))
boundary_fes2 = mesh_boundary(fes2)
cn2 = connected_nodes(bfes2)
geom2 = NodalField(fens=fens2)
T2 = NodalField(nfens=fens2.count(), dim=1)
dbc_nodes2 = np.setdiff1d(cn2, boundary_nodes2, assume_unique=True)
for index  in dbc_nodes2:
    T2.set_ebc([index], val=boundaryf(fens2.xyz[index, 0], fens2.xyz[index, 1]))
T2.apply_ebc()
femm2 = FEMMHeatDiff(material=m, fes=fes2, integration_rule=TriRule(npts=3))

T2.numberdofs()
fi2 = ForceIntensity(magn=lambda x, J: Q)
F2 = femm2.distrib_loads(geom2, T2, fi2, 3)
F2 += femm2.nz_ebc_loads_conductivity(geom2, T2)
K2 = femm2.conductivity(geom2, T2)
interface_fe_idx2 = np.unique(np.hstack([fe_select(fens2, boundary_fes2, box=box) for box in boxes]))
########################################################################################################################
# subdomain 3
########################################################################################################################
N3 = 20
fens3, fes3 = t3_ablock(0.5, 0.5, N3, N3)
fens3.xyz[:, :] += 0.5
bfes3 = mesh_boundary(fes3)
boundary_nodes3 = np.unique(np.hstack([fenode_select(fens3, box) for box in boxes ]))
boundary_fes3 = mesh_boundary(fes3)
cn3 = connected_nodes(bfes3)
geom3 = NodalField(fens=fens3)
T3 = NodalField(nfens=fens3.count(), dim=1)
dbc_nodes3 = np.setdiff1d(cn3, boundary_nodes3, assume_unique=True)
for index  in dbc_nodes3:
    T3.set_ebc([index], val=boundaryf(fens3.xyz[index, 0], fens3.xyz[index, 1]))
T3.apply_ebc()
femm3 = FEMMHeatDiff(material=m, fes=fes3, integration_rule=TriRule(npts=3))
T3.numberdofs()
fi3 = ForceIntensity(magn=lambda x, J: Q)
F3 = femm3.distrib_loads(geom3, T3, fi3, 3)
F3 += femm3.nz_ebc_loads_conductivity(geom3, T3)
K3 = femm3.conductivity(geom3, T3)
interface_fe_idx3 = np.unique(np.hstack([fe_select(fens3, boundary_fes3, box=box) for box in boxes]))
########################################################################################################################
# subdomain 4
########################################################################################################################
N4 = 40
fens4, fes4 = t3_ablock(0.5, 0.5, N4, N4)
fens4.xyz[:, 1] += 0.5
bfes4 = mesh_boundary(fes4)
boundary_nodes4 = np.unique(np.hstack([fenode_select(fens4, box) for box in boxes ]))
boundary_fes4 = mesh_boundary(fes4)
cn4 = connected_nodes(bfes4)
geom4 = NodalField(fens=fens4)
T4 = NodalField(nfens=fens4.count(), dim=1)
dbc_nodes4 = np.setdiff1d(cn4, boundary_nodes4, assume_unique=True)
for index  in dbc_nodes4:
    T4.set_ebc([index], val=boundaryf(fens4.xyz[index, 0], fens4.xyz[index, 1]))
T4.apply_ebc()
femm4 = FEMMHeatDiff(material=m, fes=fes4, integration_rule=TriRule(npts=3))
T4.numberdofs()
fi4 = ForceIntensity(magn=lambda x, J: Q)
F4 = femm4.distrib_loads(geom4, T4, fi4, 3)
F4 += femm4.nz_ebc_loads_conductivity(geom4, T4)
K4 = femm4.conductivity(geom4, T4)
interface_fe_idx4 = np.unique(np.hstack([fe_select(fens4, boundary_fes4, box=box) for box in boxes]))
########################################################################################################################
# interface 1
########################################################################################################################
mu1 =  NodalField(nfens=fens_i1.count(), dim=1)
geom_i1 = NodalField(fens=fens_i1)
mu1.numberdofs()
femm_i1 = FEMMHeatDiff(fes = fes_i1, material=m, integration_rule=GaussRule(dim=1, order=2))
M1 = femm_i1.mass(geom_i1, mu1)

g1a = assemble_gamma(fens1, boundary_fes1, interface_fe_idx1, fens_i1)
g1b = assemble_gamma(fens2, boundary_fes2, interface_fe_idx2, fens_i1)
B1a = M1 @ g1a
B1b = -M1 @ g1b
# remove dbc_nodes columns
B1a = np.delete(B1a, dbc_nodes1, axis=1)
B1b = np.delete(B1b, dbc_nodes2, axis=1)
########################################################################################################################
# interface 2
########################################################################################################################
mu2 =  NodalField(nfens=fens_i2.count(), dim=1)
geom_i2 = NodalField(fens=fens_i2)
mu2.numberdofs()
femm_i2 = FEMMHeatDiff(fes = fes_i2, material=m, integration_rule=GaussRule(dim=1, order=2))
M2 = femm_i2.mass(geom_i2, mu2)

g2a = assemble_gamma(fens2, boundary_fes2, interface_fe_idx2, fens_i2)
g2b = assemble_gamma(fens3, boundary_fes3, interface_fe_idx3, fens_i2)
B2a = M2 @ g2a
B2b = -M2 @ g2b

B2a = np.delete(B2a, dbc_nodes2, axis=1)
B2b = np.delete(B2b, dbc_nodes3, axis=1)
########################################################################################################################
# interface 3
########################################################################################################################
mu3 =  NodalField(nfens=fens_i3.count(), dim=1)
geom_i3 = NodalField(fens=fens_i3)
mu3.numberdofs()
femm_i3 = FEMMHeatDiff(fes = fes_i3, material=m, integration_rule=GaussRule(dim=1, order=2))
M3 = femm_i3.mass(geom_i3, mu3)

g3a = assemble_gamma(fens3, boundary_fes3, interface_fe_idx3, fens_i3)
g3b = assemble_gamma(fens4, boundary_fes4, interface_fe_idx4, fens_i3)
B3a = M3 @ g3a
B3b = -M3 @ g3b

B3a = np.delete(B3a, dbc_nodes3, axis=1)
B3b = np.delete(B3b, dbc_nodes4, axis=1)
########################################################################################################################
# interface 4
########################################################################################################################
mu4 =  NodalField(nfens=fens_i4.count(), dim=1)
geom_i4 = NodalField(fens=fens_i4)
mu4.numberdofs()
femm_i4 = FEMMHeatDiff(fes = fes_i4, material=m, integration_rule=GaussRule(dim=1, order=2))
M4 = femm_i4.mass(geom_i4, mu4)

g4a = assemble_gamma(fens4, boundary_fes4, interface_fe_idx4, fens_i4)
g4b = assemble_gamma(fens1, boundary_fes1, interface_fe_idx1, fens_i4)
B4a = M4 @ g4a
B4b = -M4 @ g4b

B4a = np.delete(B4a, dbc_nodes4, axis=1)
B4b = np.delete(B4b, dbc_nodes1, axis=1)



A = bmat([
    [K1,   None, None, None, B1a.T, None,  None,  B4b.T],
    [None, K2,   None, None, B1b.T, B2a.T, None,  None],
    [None, None, K3,   None, None,  B2b.T, B3a.T, None],
    [None, None, None, K4,   None,  None,  B3b.T, B4a.T],
    [B1a,  B1b,  None, None, None,  None,  None,  None],
    [None, B2a,  B2b,  None, None,  None,  None,  None],
    [None, None, B3a,  B3b,  None,  None,  None,  None],
    [B4b,  None, None, B4a,  None,  None,  None,  None],

], format='csr')


F = np.concatenate([F1, F2, F3, F4, np.zeros(fens_i1.count()), np.zeros(fens_i2.count()), np.zeros(fens_i3.count()), np.zeros(fens_i4.count())])
U = spsolve(A, F)
T1.scatter_sysvec(U[0:K1.shape[0]])
T2.scatter_sysvec(U[K1.shape[0]:K1.shape[0]+K2.shape[0]])
T3.scatter_sysvec(U[K1.shape[0]+K2.shape[0]:K1.shape[0]+K2.shape[0]+K3.shape[0]])
T4.scatter_sysvec(U[K1.shape[0]+K2.shape[0]+K3.shape[0]:K1.shape[0]+K2.shape[0]+K3.shape[0]+K4.shape[0]])

import os
script_path = __file__
script_filename = os.path.basename(script_path)[:-3]
if not os.path.exists(script_filename):
    os.mkdir(script_filename)


exact =  lambda x: 1.0 + x[0] ** 2 + 2 * x[1] ** 2
L2_err1 = L2_err(femm1, geom1, T1, exact)
L2_err2 = L2_err(femm2, geom2, T2, exact)
L2_err3 = L2_err(femm3, geom3, T3, exact)
L2_err4 = L2_err(femm4, geom4, T4, exact)

vtkexport(f"{script_filename}/left_bottom", fes1, geom1, {"temp":T1, "err":L2_err1})
vtkexport(f"{script_filename}/right_bottom", fes2, geom2, {"temp":T2, "err":L2_err2})
vtkexport(f"{script_filename}/right_top", fes3, geom3, {"temp":T3, "err":L2_err3})
vtkexport(f"{script_filename}/left_top", fes4, geom4, {"temp":T4, "err":L2_err4})
