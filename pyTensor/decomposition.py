import warnings
from typing import Iterable, List
import numpy as np
from scipy.linalg import inv
from pyTensor.tensorclass import Tensor, khatri_rao_list, rs_unfold, superdiagonal_tensor, ttl
from rpy2.robjects.packages import importr
rtensor = importr('rTensor')

""" Typiquement, pour un ndarray (n, m, l):
    num_modes = 3
    modes = (n, m, l)
    data = np.array(...)
"""


def norm(array: np.ndarray):
    # Par défaut "o" retourne la plus grande somme absolue par colonne
    # Argument "f" donne la norme de Frobenius comme pour fnorm()
    # warnings.warn("La fonction norme ne prend en compte que la norme de Frobenius pour le moment.")
    return fnorm(array)


# Vérfiée et testée pour matrices et tenseur
def fnorm(array: np.ndarray):
    return np.sqrt(np.sum(array**2))  # voir axe de sommation


def rep(val: int | float, num_rep: int):
    return [val] * num_rep


def hadamard_list(L: List|np.ndarray):
    # verifier que chacun est un vecteur ou matrice

    retmat = L[1]
    # probablement remplaçable par np.prod ou un truc du genre
    for i in range(2, len(L)):
        retmat *= L[i]

    return retmat


""" Décorticage des fonctions:
    U_list = [NULL, NULL, NULL]
    unfolded_mat = [NULL, NULL, NULL]
"""


def norm_vec(vec):
    return norm(np.asarray(vec))


def cp(tnsr: Tensor, num_components: int, max_iter=25, tol=1e-5):
    if not num_components: raise ValueError("num_components must be specified")
    if not isinstance(tnsr, Tensor): raise ValueError("'tnsr' must an instance of the Tensor class.")
    if tnsr.is_zero_tensor(): return ValueError("Zero tensor detected")

    num_modes = tnsr.num_modes
    modes = tnsr.modes
    U_list = [None] * num_modes  # Présume que num_modes est une liste, ou un np.array
    unfolded_mat = [None] * num_modes  # La même chose ?
    tnsr_norm = fnorm(tnsr)

    for m in range(num_modes):
        unfolded_mat[m] = rs_unfold(tnsr, m=m).data  # rs_unfold renvoie un tenseur
        U_list[m] = np.random.normal(loc=0.0, scale=1.0, size=(modes[m], num_components))

    est = tnsr.data
    curr_iter = 1
    converged = False

    # set up converged check

    fnorm_resid = np.zeros(max_iter)

    def check_conv(est: Tensor):
        curr_resid = fnorm(est - tnsr)
        fnorm_resid[
            curr_iter] = curr_resid  # <= pour modifier dans la fonction parent
        # https://stackoverflow.com/questions/2628621/how-do-you-use-scoping-assignment-in-r

        if curr_iter == 1: return False
        if abs(curr_resid - fnorm_resid[curr_iter - 1]) / tnsr_norm < tol:
            return True
        else:
            return False

    while curr_iter < max_iter and not converged:
        for m in range(num_modes):
            U_list_sans_m = [item for i, item in enumerate(U_list) if i != m]
            # temp_list = [U_list_sans_m[i].T @ U_list_sans_m[i] for i in range(0, m) if i !=m] # ou
            temp_list = np.array(list(map(lambda x: np.dot(np.transpose(x), x), U_list_sans_m)))
            V = hadamard_list(temp_list)
            V_inv = np.linalg.pinv(V)
            test_k = khatri_rao_list(U_list_sans_m, reverse=True)
            tmp = np.dot(unfolded_mat[m], np.dot(test_k, V_inv))

            lambdas = np.apply_along_axis(norm_vec, axis=0, arr=tmp)
            U_list[m] = np.divide(tmp, lambdas[np.newaxis, :])

            Z = superdiagonal_tensor(num_modes, num_components, lambdas)
            est = ttl(Z, U_list, ms=list(range(num_modes)))
        if check_conv(est):
            converged = True
        else:
            curr_iter += 1

    fnorm_resid = fnorm_resid[fnorm_resid != 0]
    norm_percent = (1 - (fnorm_resid[-1] / tnsr_norm)) * 100

    results = {
        "lambdas": lambdas,
        "U": U_list,
        "conv": converged,
        "est": est,
        "norm_percent": norm_percent,
        "fnorm_resid": fnorm_resid[-1],
        "all_resids": fnorm_resid
    }
    return results


"""
fonction cp, à faire :

    - Si tnsr n'est pas en tenseur, essayer de le transformer en tenseur.
    - 


"""
