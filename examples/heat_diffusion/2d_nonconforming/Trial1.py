import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.quadrilaterals import q4_blockx
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
import numpy
from numpy import array
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import reverse_cuthill_mckee
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]))



xs1 = numpy.linspace(0.0, 1.0, 2)
ys1 = numpy.linspace(0.0, 2.0, 3)
fens1, fes1 = q4_blockx(xs1, ys1)

xs2 = numpy.linspace(1.0, 2.0, 2)
ys2 = numpy.linspace(0.0, 2.0, 4)
fens2, fes2 = q4_blockx(xs2, ys2)

geom1 = NodalField(fens=fens1)
temp1 = NodalField(nfens=fens1.count(), dim=1)


def get_common_interface():
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
    