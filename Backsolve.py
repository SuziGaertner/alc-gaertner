import numpy as np

def retro(A, b):
    n = len(b)
    x = np.zeros(n)
    
    if not np.allclose(A, np.triu(A)):
        print("\nErro: A matriz não é triangular superior!")
        exit()

    if np.any(np.diag(A) == 0):  
        return "Erro: A matriz possui um " \
        "elemento nulo na diagonal principal."

    for i in range(n - 1, -1, -1):  # começa da última linha e vai subindo
        soma = 0
        for j in range(i + 1, n):
            soma += A[i, j] * x[j]
        
            x[i] = (b[i] - soma) / A[i, i]

    return x

print("Este é um programa de substituição regressiva para resolução de matrizes triangulares superiores")
linhas = int(input("Digite o número de linhas da matriz: "))

print(f"Digite os valores das {linhas} linhas separados por espaços:")

matriz = []
for i in range(linhas):
    linha = input(f"Linha {i+1}: ").split()
    linha = [float(valor) for valor in linha]  # converte os valores para float
    matriz.append(linha)

A = np.array(matriz, dtype=float)

print("\nMatriz inserida:")
print(A)

linhas_b = 1

print(f"Digite os valores da {linhas_b} linha do vetor separados por espaços:")

linha_b = input(f"Linha: ").split()
b = np.array(linha_b, dtype=float)

print("\nVetor inserido:")
print(b)

X =retro(A,b)
print("\nA resolução é X:")
print(X)