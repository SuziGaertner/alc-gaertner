import numpy as np

def is_orthogonal_by_definition(M, tol=1e-4):
    """
    Verifica se M é ortogonal pelo critério M^T M = I.
    Retorna True se M^T M for aproximadamente a matriz identidade.
    """
    if M.shape[0] != M.shape[1]:
        return False
    I = np.eye(M.shape[0])
    return np.allclose(M.T @ M, I, atol=tol)

def is_orthogonal_by_vectors(M, tol=1e-4):
    """
    Verifica se M é ortogonal checando comprimentos unitários e ortogonalidade
    entre as colunas.
    Retorna True se todas as colunas tiverem norma 1 e forem mutuamente ortogonais.
    """
    if M.shape[0] != M.shape[1]:
        return False
    cols = [M[:, i] for i in range(M.shape[1])]
    # Checa norma unitária de cada coluna
    for v in cols:
        if not np.allclose(np.dot(v, v), 1.0, atol=tol):
            return False
    # Checa ortogonalidade mútua
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if not np.allclose(np.dot(cols[i], cols[j]), 0.0, atol=tol):
                return False
    return True

# Matrizes a testar 6.38
P1 = np.array([
    [-0.40825,  0.43644,  0.80178],
    [-0.81650,  0.21822, -0.53452],
    [-0.40825, -0.87287,  0.26726]
])

P2 = np.array([
    [-0.51450,  0.48507,  0.70711],
    [-0.68599, -0.72761,  0.00000],
    [ 0.51450, -0.48507,  0.70711]
])

# Matrizes a testar 6.39
P3 = np.array([
    [-0.58835, 0.70206, 0.40119],
    [-0.78446, -0.37524, -0.49377],
    [-0.19612, -0.60523, 0.77152]
])

P4 = np.array([
    [-0.47624, -0.4264, 0.30151],
    [0.087932, 0.86603, -0.40825],
    [-0.87491, -0.26112, 0.86164]
])


# Teste das funções com tolerância ajustada
for name, P in [('P1', P1), ('P2', P2),('P3', P3),('P4', P4)]:
    print(f"Testando {name}:")
    print(" - Ortogonal por definição?       ", is_orthogonal_by_definition(P))
    print(" - Ortogonal verificando vetores? ", is_orthogonal_by_vectors(P))