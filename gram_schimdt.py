import numpy as np

def mod_gram_schmidt(A):
    m, n = A.shape

    Q = A.copy().astype(np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], Q[:, i])
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R

# [FORD] Exercício 14.15
A = np.array([
    [1, 9, 0, 5, 3, 2],
    [-6, 3, 8, 2, -8, 0],
    [3, 15, 23, 2, 1, 7],
    [3, 57, 35, 1, 7, 9],
    [3, 5, 6, 15, 55, 2],
    [33, 7, 5, 3, 5, 7]
], dtype=float)

Q, R = mod_gram_schmidt(A)

print("Matriz A:")
print(A)

print("\n--- Base Ortonormal (Colunas da Matriz Q) ---")
print("As colunas da matriz Q abaixo formam uma base ortonormal para o espaço gerado pelas colunas de A:")
print(np.round(Q, 5))

print("\nMatriz R (triangular superior):")
print(np.round(R, 5))


# Verificação das propriedades:
print("\nVerificação se A ≈ QR:", np.allclose(A, Q @ R))
print("Verificação se Q^T Q ≈ I:", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))

# Executa a decomposição QR com a função do NumPy
Q_np, R_np = np.linalg.qr(A)

print("\n" + "="*50)
print("COMPARAÇÃO DOS RESULTADOS")
print("="*50 + "\n")

# --- Resultados do Gram-Schmidt Modificado ---
print("--- Resultados do Gram-Schmidt Modificado ---")
print("Matriz Q (MGS):")
print(np.round(Q, 5))
print("\nMatriz R (MGS):")
print(np.round(R, 5))

# --- Resultados da implementação do NumPy ---
print("\n--- Implementação do NumPy (numpy.linalg.qr) ---")
print("Matriz Q (NumPy):")
print(np.round(Q_np, 5))
print("\nMatriz R (NumPy):")
print(np.round(R_np, 5))

print("\n" + "="*50)
print("VERIFICAÇÃO DAS PROPRIEDADES")
print("="*50 + "\n")

# 1. Verificação da Reconstrução: A = QR
reconstruction_mgs_ok = np.allclose(Q @ R, A)
reconstruction_np_ok = np.allclose(Q_np @ R_np, A)

print(f"Reconstrução A = QR (Função Gram-Schmidt Modificado): {reconstruction_mgs_ok}")
print(f"Reconstrução A = QR (Função NumPy): {reconstruction_np_ok}")

# 2. Verificação da Ortonormalidade: Q.T @ Q = I
orthonormality_mgs_ok = np.allclose(Q.T @ Q, np.identity(A.shape[1]))
orthonormality_np_ok = np.allclose(Q_np.T @ Q_np, np.identity(A.shape[1]))

print(f"\nOrtonormalidade de Q (Função Gram-Schmidt Modificado): {orthonormality_mgs_ok}")
print(f"Ortonormalidade de Q (Função NumPy): {orthonormality_np_ok}")

# 3. Comparação direta ignorando os sinais
# Comparamos os valores absolutos para ver se as bases são essencialmente as mesmas
q_abs_close = np.allclose(np.abs(Q), np.abs(Q_np))
r_abs_close = np.allclose(np.abs(R), np.abs(R_np))

print(f"\nOs valores absolutos das matrizes Q são próximos? {q_abs_close}")
print(f"Os valores absolutos das matrizes R são próximos? {r_abs_close}")