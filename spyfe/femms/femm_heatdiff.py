import numpy
from numpy import dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_base import FEMMBase
from spyfe.csys import CSys
import scipy.sparse


class FEMMHeatDiff(FEMMBase):
    """
    The class is for operations on models of heat conduction.

    """

    def __init__(self, fes=None, integration_rule=None, material_csys=CSys(), assoc_geom=None,
                 material=None):
        """Constructor.

        :param material: Material object.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        super().__init__(fes=fes, integration_rule=integration_rule,
                         material_csys=material_csys, assoc_geom=assoc_geom)
        self.material = material

    def conductivity(self, geom, temp):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        kappa_bar = self.material.thermal_conductivity  # in local material coordinate
        mcsmtx = None
        mcs = self.material_csys
        if mcs.isconstant:
            mcsmtx = mcs.eval_matrix()  # constant
        assm = SysmatAssemblerSparseFixedSymm(fes, temp)
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                if not mcs.isidentity:  # Transformation is required
                    if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                        mcsmtx = mcs.eval_matrix(dot(bfuns[j].T, x), jacmat, fes.label[i])
                    fes.gradbfun(gradbfun, gradbfunpars[j], mcsmtx.T * jacmat)
                else:  # material coordinates same as global coordinates
                    fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                assm.elmtx[i, :, :] += dot(gradbfun, dot((jac * w[j]) * kappa_bar, gradbfun.T))
        return assm.make_matrix()

    def mass(self, geom, temp):
        """Compute the mass matrix.

        :param geom: Geometry field.
        :param temp: temperature field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradfunpars, npts, pc, w = self.integration_data()
        assm = SysmatAssemblerSparseFixedSymm(fes, temp)
        nexp_nexp = []  # Precomputed for efficiency
        for j in range(npts):
            nexp = numpy.zeros((temp.dim, assm.elem_mat_nrowcol))
            for m in range(fes.nfens):
                nexp[:, m * temp.dim:(m + 1) * temp.dim] = numpy.identity(temp.dim) * bfuns[j][m]
            nexp_nexp.append(dot(nexp.T, nexp))
        rho = self.material.rho
        jacmat = numpy.zeros((geom.dim, fes.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                assm.elmtx[i, :, :] += nexp_nexp[j] * (rho * jac * w[j])
        return assm.make_matrix()
    def mass_mortar(self, geom, temp):
        """Compute the mass matrix.

        :param geom: Geometry field.
        :param temp: temperature field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradfunpars, npts, pc, w = self.integration_data()
        assm = SysmatAssemblerSparseFixedSymm(fes, temp)
        nexp_nexp = []  # Precomputed for efficiency
        bfuns_left = [numpy.array([[1],[1]]), numpy.array([[0.21132487], [0.78867513]])]
        bfuns_right = [numpy.array([[0.78867513],[0.21132487]]), numpy.array([[1],[1]])]
        nexp_nexp_left =[]
        nexp_nexp_right =[]
        # bfuns_endpts = [1,1]
        for j in range(npts):
            nexp = numpy.zeros((temp.dim, assm.elem_mat_nrowcol))
            nexpl = numpy.zeros((temp.dim, assm.elem_mat_nrowcol))
            nexpr = numpy.zeros((temp.dim, assm.elem_mat_nrowcol))
            for m in range(fes.nfens):
                nexp[:, m * temp.dim:(m + 1) * temp.dim] = numpy.identity(temp.dim) * bfuns[j][m]
                nexpl[:, m * temp.dim:(m + 1) * temp.dim] = numpy.identity(temp.dim) * bfuns_left[j][m]
                nexpr[:, m * temp.dim:(m + 1) * temp.dim] = numpy.identity(temp.dim) * bfuns_right[j][m]

            nexp_nexp.append(dot(nexp.T, nexp))
            nexp_nexp_left.append(dot(nexpl.T, nexpl))
            nexp_nexp_right.append(dot(nexpr.T, nexpr))
        rho = self.material.rho
        jacmat = numpy.zeros((geom.dim, fes.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                if i==0:
                    assm.elmtx[i, :, :] += nexp_nexp_left[j] * (rho * jac * w[j])
                elif i==fes.conn.shape[0]-1:
                    assm.elmtx[i, :, :] += nexp_nexp_right[j] * (rho * jac * w[j])
                else:
                    assm.elmtx[i, :, :] += nexp_nexp[j] * (rho * jac * w[j])
        return assm.make_matrix()
    def lam_mat(self, geom, temp):
        """Compute the lam matrix. works for only 1 dim for now

        :param geom: Geometry field.
        :param temp: temperature field.
        :return: Sparse matrix.

        """
        fes = self.fes
        bfuns, gradfunpars, npts, pc, w = self.integration_data()
        mat = scipy.sparse.csr_matrix((fes.count()*temp.dim, fes.count()*temp.dim+1))
        assm = SysmatAssemblerSparseFixedSymm(fes, temp)
        nexp_ = []  # Precomputed for efficiency
        for j in range(npts):
            nexp = numpy.zeros((temp.dim, assm.elem_mat_nrowcol))
            for m in range(fes.nfens):
                nexp[:, m * temp.dim:(m + 1) * temp.dim] = numpy.identity(temp.dim) * bfuns[j][m]
            nexp_.append(nexp)
        rho = self.material.rho
        jacmat = numpy.zeros((geom.dim, fes.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                mat[i, i:i+2] += nexp_[j] * (rho * jac * w[j])
        return mat



    def nz_ebc_loads_conductivity(self, geom, temp):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        kappa_bar = self.material.thermal_conductivity  # in local material coordinate
        mcsmtx = None
        mcs = self.material_csys
        if mcs.isconstant:
            mcsmtx = mcs.eval_matrix()  # constant
        conns = fes.conn  # input connectivity
        assm = SysvecAssembler(fes, temp)
        ke = numpy.zeros((assm.elem_mat_nrowcol, assm.elem_mat_nrowcol))
        jacmat = numpy.zeros((geom.dim, geom.dim))
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        pt = numpy.zeros((self.fes.conn.shape[1] * temp.dim,), dtype=numpy.float64)
        for i in range(conns.shape[0]):
            conn = conns[i, :]
            temp.gather_fixed_values_vec(conn, pt)
            if any(pt != 0.0):
                x = geom.values[conn, :]
                ke.fill(0.0)
                for j in range(npts):
                    jacmat = dot(x.T, gradbfunpars[j])
                    jac = fes.jac_volume(conn, bfuns[j], jacmat, x)
                    if not mcs.isidentity:  # Transformation is required
                        if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                            mcsmtx = mcs.eval_matrix(dot(bfuns[j].T, x), jacmat, fes.label[i])
                        fes.gradbfun(gradbfun, gradbfunpars[j], mcsmtx.T * jacmat)
                    else:  # material coordinates same as global coordinates
                        fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                    ke += dot(gradbfun, dot((jac * w[j]) * kappa_bar, gradbfun.T))
                assm.elvec[i, :] = -dot(ke, pt).ravel()
        return assm.make_vector()

    def distrib_loads(self, geom, temp, fi, m):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        bfunts = list()
        for index in range(len(bfuns)):
            bfunts.append(bfuns[index].T)
            bfunts[index] = bfunts[index].ravel()
            bfuns[index] = bfuns[index].ravel()
        jacmat = numpy.zeros((geom.dim, geom.dim))
        assm = SysvecAssembler(fes, temp)
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac = fes.jac_mdim(fes.conn[i, :], bfuns[j], jacmat, x, m)
                f = fi.get_magn(dot(bfunts[j], x), jacmat)
                assm.elvec[i, :] += (bfuns[j] * (f * jac * w[j]))
        return assm.make_vector()
