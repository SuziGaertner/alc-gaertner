import numpy as np
from scipy.linalg import lu


def ludecomp(A):
    A = A.copy().astype(float)
    n = A.shape[0] #Verifica se a matriz é quadrada
    if A.shape[1] != n:
        raise ValueError("A matriz de entrada deve ser quadrada.")

    L = np.eye(n)
    P = np.eye(n)

    for i in range(n - 1):
        # Pivoteamento parcial
        max_row = np.argmax(abs(A[i:n, i])) + i
        if max_row != i:
            # Trocar linhas em A (somente da coluna i até o fim)
            A[[i, max_row], i:] = A[[max_row, i], i:]
            # Trocar linhas em P (linha inteira)
            P[[i, max_row], :] = P[[max_row, i], :]
            # Trocar linhas em L (somente nas colunas anteriores a i)
            L[[i, max_row], :i] = L[[max_row, i], :i]

        # Calcula os multiplicadores
        multipliers = A[i + 1:n, i] / A[i, i]
        L[i + 1:n, i] = multipliers

        # Atualiza a submatriz
        A[i + 1:n, i + 1:n] = A[i + 1:n, i + 1:n] - np.outer(multipliers, A[i, i + 1:])

        # Zera explicitamente abaixo do pivô
        A[i + 1:n, i] = 0.0

    U = A
    return L, U, P

# Exemplo de teste
#np.random.seed(0)
A =  np . array ([[2 , 1 , 5] , [ 4 , 4 , -4] , [ 1 , 3 , 1]] , dtype = float )

# Usando a implementação própria
L1, U1, P1 = ludecomp(A)

# Usando scipy.linalg.lu
P2, L2, U2 = lu(A)

print("Matriz A:")
print(A)
print("\n--- Implementação própria ---")
print("P:")
print(P1)
print("L:")
print(L1)
print("U:")
print(U1)

print("\n--- Scipy LU ---")
print("P:")
print(P2)
print("L:")
print(L2)
print("U:")
print(U2)

# Verificação se PA = LU
print("\nVerificação da implementação própria:")
print("PA ≈ LU ?", np.allclose(P1 @ A, L1 @ U1))

print("\nVerificação da Scipy:")
print("A ≈ PLU ?", np.allclose(A, P2 @ L2 @ U2))