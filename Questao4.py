import numpy as np
import pandas as pd
import math

def compute_norm(vec):
    """Calcula a norma L2 de um vetor sem usar numpy.linalg."""
    total = 0.0
    for val in vec.flatten():
        total += val * val
    return math.sqrt(total)

def main():
    resultados = []
    for n in [5, 15, 25]:
        # Gera u e v em R^n
        u = np.random.rand(n, 1)
        v = np.random.rand(n, 1)
        # Constrói A = u v^T
        A = u @ v.T

        # Calcula normas de u e v passo a passo
        norm_u = compute_norm(u)
        norm_v = compute_norm(v)

        # Determina rank(A): para matriz uv^T, rank = 1 se u e v ≠ 0, senão 0
        if norm_u == 0.0 or norm_v == 0.0:
            rank_A = 0
        else:
            rank_A = 1

        # Produto das normas
        produto_normas = norm_u * norm_v

        # Norma espectral de A (posto 1) = produto_normas
        norma_espectral = produto_normas

        resultados.append({
            'n': n,
            'rank(uv^T)': rank_A,
            '||u||₂·||v||₂': produto_normas,
            '||uvᵀ||₂': norma_espectral
        })

    df = pd.DataFrame(resultados)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()