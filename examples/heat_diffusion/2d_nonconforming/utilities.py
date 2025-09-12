import numpy as np

from spyfe.fields.elemental_field import ElementalField


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
        if np.abs(cross) < 1e-8 and 0 <= dot <= np.dot(ab, ab):
            return True
        return False
    # elif element_dim == 2:
    #     path = Path(element)
    #     return path.contains_point(node)
    else:
        raise NotImplementedError("Only 1D elements are supported.")


def compute_xi(node_xyz, element_xyz, element_dim=1):
    """
    Compute the isoparametric coordinates (xi) of a node within an element.
    For 1D: maps node position to local coordinate in [-1, 1].
    For 2D: (not implemented).
    """
    node = np.array(node_xyz)
    element = np.array(element_xyz)
    if element_dim == 1:
        a, b = element
        # Map node position to local coordinate xi in [-1, 1]
        length = np.linalg.norm(b - a)
        if length == 0:
            raise ValueError("Element has zero length.")
        xi = 2 * np.dot(node - a, b - a) / np.dot(b - a, b - a) - 1
        return np.array([xi])
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