import numpy as np

def norma_p_vetor(vetor, p):
    if p == float('inf'):
        return np.max(np.abs(vetor))
    return np.sum(np.abs(vetor)**p)**(1/p)

def norma_p_matriz_2por2(matriz: np.array, p: float) -> float:
    if not isinstance(matriz, np.ndarray):
        matriz = np.array(matriz, dtype=float)
    
    if matriz.shape != (2, 2):
        raise ValueError("A matriz deve ser 2x2.")
    
    if p < 1 and p != float('inf'):
        raise ValueError("p deve ser >= 1 ou float('inf').")
    
    if p == 1:
        # Norma 1: maior soma absoluta das colunas
        return np.max(np.sum(np.abs(matriz), axis=0))
    
    if p == 2:
        # Norma 2 induzida: maior valor singular (raiz do maior autovalor de A^T A)
        ata = np.dot(matriz.T, matriz)
        traco = ata[0, 0] + ata[1, 1]
        det = ata[0, 0] * ata[1, 1] - ata[0, 1] * ata[1, 0]
        discrim = traco**2 - 4 * det
        discrim = max(discrim, 0)
        max_lambda = (traco + np.sqrt(discrim)) / 2
        return np.sqrt(max_lambda)

    if p == float('inf'):
        # Norma infinito: maior soma absoluta das linhas
        return np.max(np.sum(np.abs(matriz), axis=1))
    
    # Estimativa numérica para outros p
    samples = 30000
    max_ratio = 0.0

    for i in range(samples):
        x = np.random.randn(2)
        n_x = norma_p_vetor(x, p)
        if n_x < 1e-10:
            continue
        ax = np.dot(matriz, x)
        n_ax = norma_p_vetor(ax, p)
        ratio = n_ax / n_x
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio
