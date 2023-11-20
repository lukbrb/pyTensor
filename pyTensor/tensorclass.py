import warnings
from typing import List
import numpy as np
from scipy import linalg


class Tensor(np.ndarray):
    def __new__(cls, num_modes: int, modes: tuple[int], data: np.ndarray):
        obj = np.asarray(data).view(cls)
        obj.num_modes = num_modes
        obj.modes = np.asarray(modes)
        if np.any(obj.modes) <= 0: raise ValueError(
            "'modes' must contain strictly positive values; if np.any mode is 1, consider a smaller num_modes")
        return obj

    # Pour pouvoir accéder aux données via .data et mimiquer rTensor. Mais inutile, à déprécier
    @property
    def data(self):
        return self[...]

    def is_zero_tensor(self):
        if np.sum(self.data == 0) == np.prod(np.asarray(self.modes)): return True
        return False


def astensor(array: np.ndarray) -> Tensor:
    modes = array.shape
    num_modes = len(modes)
    return Tensor(num_modes=num_modes, modes=modes, data=array)


def unfold(tensor: Tensor, row_idx: List[int], col_idx: List[int], order='F') -> Tensor:
    rs = np.asarray(row_idx)
    cs = np.asarray(col_idx)
    # if not rs or not cs: raise ValueError("row and column indices must be specified")
    num_modes = tensor.num_modes  # pour démarrer à zero

    if len(rs) + len(cs) != num_modes:
        raise ValueError("Incorrect number of indices. Number of modes not reached.")
    if np.any(rs < 0) or np.any(rs > num_modes - 1) or np.any(cs < 0) or np.any(cs > num_modes - 1):
        raise ValueError(
            "Illegal indices specified. 'row_idx' and 'col_idx' must be positive and strictly less than 'num_modes'.")
    perm = np.array(list(rs) + list(cs))
    if np.any(np.sort(perm) != np.arange(num_modes)):
        raise ValueError("Missing and/or repeated indices")

    modes = tensor.modes
    mat = tensor.data
    new_modes = np.array([np.prod(modes[rs]), np.prod(modes[cs])])
    mat = np.transpose(tensor.data, perm)  # probablement soustraire -1 à perm pour les indices
    mat = mat.reshape(new_modes, order=order)  # rearrangement style fortran comme pour dim() <- dim en R:
    # https://rstudio.github.io/reticulate/reference/array_reshape.html
    return astensor(mat)


def rs_unfold(tensor: Tensor, m: int, order='F') -> Tensor:
    assert 0 <= m < tensor.num_modes, f"'m' must be a valid mode of the tensor, not {m}."
    rs = np.asarray([m])
    cs = np.asarray([i for i in range(tensor.num_modes) if i != m])
    return unfold(tensor, row_idx=rs, col_idx=cs, order=order)


# Validé par essai manuel
def superdiagonal_tensor2(num_modes, length, elements=1):
    modes = [length] * num_modes
    arr = np.zeros(modes, dtype=np.float32)

    if isinstance(elements, int):
        elements = [elements] * length

    for i in range(length):
        indices = [i] * num_modes
        arr[tuple(indices)] = elements[i]

    return astensor(arr)


# L'implémentation originale pernd une liste en argument, et les multiplie entre elles, "element-wise".
# L'opération est largement simplifiée avec un ndarray

# Vérifiée à la main
# def hadamard_list(L: np.ndarray) -> np.ndarray:
#     # TODO: Verif forme des tableaux, et de la nature de L``
#     # return np.prod(L, axis=-1)  # typiquement axis=2 dans notre cas
#
#     retmat = L[0]
#     for matrice in L[1:]:
#         retmat = np.multiply(retmat, matrice)
#     return retmat


def hadamard_list(L):
    retmat = L[0]
    for matrice in L[1:]:
        retmat *= matrice
    return retmat


def kronecker_list(L):
    result = L[0]
    for matrix in L[1:]:
        result = np.kron(result, matrix)
    return result


def superdiagonal_tensor(num_modes, length, elements=1):
    modes = np.repeat(length, num_modes)
    arr = np.zeros(modes)
    if isinstance(elements, int) == 1:
        elements = np.repeat(elements, length)
    for i in range(length):
        txt = "arr[" + ",".join([str(i)] * num_modes) + "]=" + str(elements[i])
        txt = txt.replace(" ", ", ")
        print(txt)
        exec(txt)
    return arr


def khatri_rao_list_2(L, reverse=False):
    if reverse:
        L = L[::-1]

    retmat = L[0]
    for matrice in L[1:]:
        retmat = linalg.khatri_rao(retmat, matrice)
    return retmat


def khatri_rao_list(L, reverse=False):
    assert all([isinstance(x, np.ndarray) for x in L]), "All elements in L must be matrices"
    ncols = [x.shape[1] for x in L]
    assert len(set(ncols)) == 1, "All matrices in L must have the same number of columns"
    ncols = ncols[0]
    nrows = [x.shape[0] for x in L]
    retmat = np.zeros((np.prod(nrows), ncols))
    if reverse:
        L = L[::-1]
    for j in range(ncols):
        Lj = [x[:, j] for x in L]
        retmat[:, j] = kronecker_list(Lj)
    return retmat


def khatri_rao_list_bis(L, reverse=False):
    # Vérifie que tous les éléments de L sont des matrices
    assert all(isinstance(matrix, np.ndarray) for matrix in L), "Tous les éléments de L doivent être des matrices"

    # Vérifie que toutes les matrices ont le même nombre de colonnes
    ncols = [matrix.shape[1] for matrix in L]
    assert len(set(ncols)) == 1, "Toutes les matrices doivent avoir le même nombre de colonnes"
    ncols = ncols[0]

    # Initialise la matrice résultante
    nrows = [matrix.shape[0] for matrix in L]

    retmat = np.zeros((np.prod(nrows), ncols))
    # Inverse l'ordre des matrices si reverse=True
    if reverse:
        L = L[::-1]

    # Remplit la matrice résultante en utilisant le produit de Kronecker
    for j in range(ncols):
        # Lj = [matrix[:, j] for matrix in L]
        # retmat[:, j] = kronecker_list(Lj)
        retmat = linalg.khatri_rao(a, b)
    return retmat


def ttl(tnsr, list_mat, ms=None):
    if ms is None or not isinstance(ms, (list, np.ndarray)):
        raise ValueError("m modes must be specified as a vector")

    if len(ms) != len(list_mat):
        raise ValueError("m modes length does not match list_mat length")

    num_mats = len(list_mat)
    if len(set(ms)) != num_mats:
        print("Consider pre-multiplying matrices for the same m for speed")

    mat_nrows = [mat.shape[0] for mat in list_mat]
    mat_ncols = [mat.shape[1] for mat in list_mat]

    for i in range(num_mats):
        mat = list_mat[i]
        m = ms[i]

        mat_dims = mat.shape
        modes_in = tnsr.modes

        if modes_in[m] != mat_dims[1]:
            raise ValueError(f"Modes mismatch: tnsr.modes[{m}] != mat.shape[1]")

        modes_out = modes_in.copy()
        modes_out[m] = mat_dims[0]

        tnsr_m = rs_unfold(tnsr, m=m).data
        retarr_m = np.dot(mat, tnsr_m)
        tnsr = rs_fold(retarr_m, m=m, modes=modes_out)

    return tnsr


def fold(mat: Tensor | np.ndarray, row_idx: List[int], col_idx: List[int], modes: List[int], order='F'):
    rs = row_idx
    cs = col_idx

    if not isinstance(mat, np.ndarray):
        raise ValueError("mat must be of type 'numpy.ndarray'")
    if mat.ndim != 2:
        raise ValueError("mat must be a 2D matrix")

    num_modes = len(modes)
    if num_modes != len(rs) + len(cs):
        raise ValueError("Number of modes does not match the sum of row and column space indices")

    mat_modes = mat.shape
    if mat_modes[0] != np.prod([modes[i] for i in rs]) or mat_modes[1] != np.prod([modes[i] for i in cs]):
        raise ValueError("Matrix dimensions do not match Tensor modes")

    # iperm = [modes.index(mode) + 1 for mode in rs + cs]
    modes = list(modes)
    iperm = rs + cs
    # iperm = [modes.index(x) + 1 if x in modes else None for x in rs + cs]
    # iperm = [np.where(np.array(modes) == mode)[0][0] if mode in modes else None for mode in rs + cs]
    modes = np.asarray(modes)
    mat = mat.reshape([modes[i] for i in rs] + [modes[i] for i in cs], order=order)
    # folded_tensor = np.transpose(mat, iperm)
    folded_tensor = np.moveaxis(mat, range(len(rs) + len(cs)), rs + cs)
    # mat = mat.reshape(new_modes, order='F')  # rearrangement style fortran comme pour dim() <- dim en R:
    # https://rstudio.github.io/reticulate/reference/array_reshape.html return astensor(mat)
    return astensor(folded_tensor)


def k_fold(mat: Tensor | np.ndarray, m: int, modes: List[int], order='F') -> Tensor:
    num_modes = len(modes)
    rs = [m]
    cs = [i for i in range(num_modes) if i != m]  # vérifier si on bouge m, ou l'indice lié à m
    return fold(mat, row_idx=rs, col_idx=cs, modes=modes, order=order)


def rs_fold(mat: Tensor | np.ndarray, m: int, modes: List[int], order='F') -> Tensor:
    return k_fold(mat, m, modes, order)
