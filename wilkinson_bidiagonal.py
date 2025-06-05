import numpy as np
import matplotlib.pyplot as plt

def wilkinson_bidiagonal(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = n - i          # Diagonal principal
    for i in range(n-1):
        A[i, i+1] = n            # Diagonal superior
    return A

# Exemplo
print(wilkinson_bidiagonal(5))

cond_numbers = []
orders = range(1, 16)

for n in orders:
    A = wilkinson_bidiagonal(n)
    cond = np.linalg.cond(A)
    cond_numbers.append(cond)

plt.figure(figsize=(8,5))
plt.plot(orders, cond_numbers, marker='o')
plt.xlabel('Ordem da matriz (n)')
plt.ylabel('Número de condicionamento (cond)')
plt.title('Número de condicionamento da matriz bidiagonal de Wilkinson')
plt.grid(True)
plt.show()

# Matriz original
n = 20
A = wilkinson_bidiagonal(n)
eigvals_original = np.linalg.eigvals(A)

# Perturbação: soma 1e-10 ao elemento (20,1)
A_perturbed = A.copy()
A_perturbed[-1, 0] += 1e-10
eigvals_perturbed = np.linalg.eigvals(A_perturbed)

# Comparação dos autovalores
print("Autovalores originais:\n", np.sort(np.real(eigvals_original)))
print("\nAutovalores após perturbação:\n", np.sort(np.real(eigvals_perturbed)))
print("\nDiferença máxima:", np.max(np.abs(np.sort(np.real(eigvals_original)) - np.sort(np.real(eigvals_perturbed)))))