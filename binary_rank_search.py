import numpy as np
import tensorly as tl
from tensorly.decomposition import CP, parafac


def compute_error(tensor: tl.tensor, rank: int, n_iter_max: int = 2000000, tol: float = 1e-9) -> int:
    try:
        fac = parafac(tensor, rank=rank, n_iter_max=n_iter_max, tol=tol, linesearch=True)
        err_min = tl.norm(tl.cp_to_tensor(fac) - tensor)
    except Exception as e:
        print(f"{e} - Setting 'error' to infinity for {rank=}.")
        err_min = np.inf
    return err_min


def optimize_rank(tenseur: tl.tensor, low_rank: int, high_rank: int) -> dict:
    errors = dict()
    while (high_rank - low_rank) > 0:
        current_rank = int(np.ceil((low_rank + high_rank) / 2))

        result = compute_error(tensor=tenseur, rank=current_rank)

        print(f"{current_rank=} -> {result=}")
        errors[current_rank] = result
        print("Error:", result, "Rank:", current_rank)

        if result == np.inf:
            high_rank = current_rank - 1
        else:
            low_rank = current_rank

    # NOTE: Pas compris cette partie de ton code
    # opt_r = low_rank
    # rank_range = np.arange(1, len(errors.keys()), 1)  # fait des pas de 1 de 0 à len(errors)

    return errors


if __name__ == '__main__':
    m1 = np.loadtxt("m1.txt", dtype=np.int32)
    m2 = np.loadtxt("m2.txt", dtype=np.int32)
    tenseur = tl.tensor(np.concatenate((m1[..., None], m2[..., None]), axis=2))  # [..., None] crée un nouvel axe vide
    erreurs = optimize_rank(tenseur, 1, 100)
    print(erreurs)
    print("Rang optimal :", min(erreurs, key=erreurs.get))
