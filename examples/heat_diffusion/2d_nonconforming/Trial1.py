import os
import sys

from scipy.special import lmbda

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
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport
from spyfe.meshing.generators.intervals import l2_blockx_2D
from spyfe.meshing.selection import connected_nodes, fe_select, fenode_select



k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]), rho=1.0)



xs1 = np.linspace(0.0, 1.0, 2)
ys1 = np.linspace(0.0, 2.0, 3)
fens1, fes1 = q4_blockx(xs1, ys1)


xs2 = np.linspace(1.0, 2.0, 2)
ys2 = np.linspace(0.0, 2.0, 4)
fens2, fes2 = q4_blockx(xs2, ys2)

N=17

ys_i = np.linspace(0.0, 2.0, N)  # x-coordinates
xs_i = np.full_like(ys_i, 1.0)     # y-coordinates (constant)
fens_i, fes_i = l2_blockx_2D(xs_i, ys_i)

mu =  NodalField(nfens=fens_i.count(), dim=1)
geom_i = NodalField(fens=fens_i)
mu.numberdofs()
femm_i = FEMMHeatDiff(fes = fes_i, material=m, integration_rule=GaussRule(dim=1, order=2))
M = femm_i.mass(geom_i, mu)
M_array = M.toarray()
cond = femm_i.conductivity(geom_i, mu)

# femm_i_defor = FEMMDefor(fes)


geom1 = NodalField(fens=fens1)
temp1 = NodalField(nfens=fens1.count(), dim=1)

subdomains  = [[fes1, fens1], [fes2, fens2]]












###### just for structure ####################

def get_common_interface(subdomains):
#     sub_boundaries = [mesh_boundary(subdomain[0]) for subdomain in subdomains]
#     for i in range(len(sub_boundaries)):
#        for j in range(len(sub_boundaries)):
#            if j > i:
#                pass
#
#
#
    pass


def mesh_common_interface():
    pass


def create_big_matrix():
    pass


def create_B1():
    pass


def assemble_big_matrix():
    pass


def assemble_big():
    get_common_interface()
    mesh_common_interface()
    create_big_matrix()
    create_B1()
    assemble_big_matrix()
    