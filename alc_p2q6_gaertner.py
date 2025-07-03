import numpy as np

def multiplicacao_matrizes(A, B):
    n_linhas_A, n_colunas_A = A.shape
    n_linhas_B, n_colunas_B = B.shape

    if n_colunas_A != n_linhas_B:
        raise Exception("As dimensões das matrizes não permitem multiplicação.")

    # Cria uma matriz de zeros para o resultado com as dimensões corretas.
    C = np.zeros((n_linhas_A, n_colunas_B))

    for i in range(n_linhas_A):
        for j in range(n_colunas_B):
            soma = 0
            for k in range(n_colunas_A):
                soma += A[i, k] * B[k, j]
            C[i, j] = soma
    return C

def matriz_inversa(A):
    n = A.shape[0]
    
    # Cria uma cópia para não modificar a matriz original
    matriz_A = A.copy()

    # Cria uma matriz identidade do mesmo tamanho
    identidade = np.zeros((n, n))
    for i in range(n):
        identidade[i, i] = 1.0

    # Processo de Eliminação de Gauss-Jordan
    for i in range(n):
        # Encontra o pivô (maior valor na coluna para estabilidade numérica)
        pivo_linha = i
        for j in range(i + 1, n):
            if abs(matriz_A[j, i]) > abs(matriz_A[pivo_linha, i]):
                pivo_linha = j
        
        # Troca a linha atual pela linha do pivô
        matriz_A[[i, pivo_linha]] = matriz_A[[pivo_linha, i]]
        identidade[[i, pivo_linha]] = identidade[[pivo_linha, i]]

        # Normaliza a linha do pivô para que o elemento da diagonal seja 1
        divisor = matriz_A[i, i]
        if divisor == 0:
            raise Exception("A matriz não é invertível (ou é mal condicionada).")
            
        matriz_A[i, :] /= divisor
        identidade[i, :] /= divisor

        # Zera os outros elementos da coluna
        for j in range(n):
            if i != j:
                fator = matriz_A[j, i]
                matriz_A[j, :] -= fator * matriz_A[i, :]
                identidade[j, :] -= fator * identidade[i, :]

    return identidade

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def aproximacao_truncada(A, m):
    A = np.array(A)
    # --- 1. Validação das Entradas ---
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise Exception("Erro: A matriz de entrada não é quadrada.")
    
    n = A.shape[0]

    if not (0 <= m <= n):
        raise Exception("Erro: O número de autovalores a serem mantidos (m) deve estar entre 0 e a dimensão da matriz.")

    # --- 2. Decomposição (Função Permitida) ---
    autovalores, autovetores = np.linalg.eig(A)

    # --- 3. Seleção dos Autovalores (Lógica Manual) ---
    # Calcula os valores absolutos (Função Permitida)
    abs_autovalores = np.abs(autovalores)

    # Cria uma lista de tuplas (valor_absoluto, indice_original) para ordenar
    autovalores_com_indices = []
    for i in range(n):
        autovalores_com_indices.append((abs_autovalores[i], i))
    
    # Ordena a lista em ordem decrescente com base no valor absoluto
    autovalores_com_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Cria um novo vetor de autovalores, inicialmente com zeros (Função Permitida)
    autovalores_filtrados = np.zeros(n, dtype=float)
    
    # Preenche o novo vetor com os m maiores autovalores nas posições corretas
    for i in range(m):
        indice_original = autovalores_com_indices[i][1]
        autovalores_filtrados[indice_original] = autovalores[indice_original]

    # --- 4. Criação da Matriz Diagonal (Lógica Manual) ---
    D_truncado = np.zeros((n, n), dtype=float)
    for i in range(n):
        D_truncado[i, i] = autovalores_filtrados[i]

    # --- 5. Reconstrução da Matriz A' (Usando Funções Auxiliares) ---
    # A' = V * D' * V⁻¹
    V = autovetores
    V_inv = matriz_inversa(V)
    
    # A_truncada = V @ D_truncado @ V_inv
    temp = multiplicacao_matrizes(V, D_truncado)
    A_truncada = multiplicacao_matrizes(temp, V_inv)
    
    # Retorna a parte real para evitar pequenos ruídos numéricos imaginários
    return np.real(A_truncada)