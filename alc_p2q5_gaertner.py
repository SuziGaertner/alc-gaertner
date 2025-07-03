import numpy as np

def produto_escalar(v1, v2):
    # Produto escalar: soma dos produtos dos componentes correspondentes
    # Ex: [a,b,c] . [x,y,z] = ax + by + cz
    return sum(v1_comp * v2_comp for v1_comp, v2_comp in zip(v1, v2))


def volume_vetores(vetor1, vetor2, vetor3):
    # 1. Converter os vetores para arrays NumPy
    v1 = np.array(vetor1)
    v2 = np.array(vetor2)
    v3 = np.array(vetor3)

    # 2. Validar se todos os vetores têm a mesma dimensão (comprimento)
    if not (len(v1) == len(v2) == len(v3)):
        return "Erro: Todos os vetores devem ter a mesma dimensão (número de componentes)."

    # 3. Construir a Matriz de Gram (G)
    # G_ij = vetor_i . vetor_j
    g11 = produto_escalar(v1, v1)
    g12 = produto_escalar(v1, v2)
    g13 = produto_escalar(v1, v3)

    g21 = produto_escalar(v2, v1)
    g22 = produto_escalar(v2, v2)
    g23 = produto_escalar(v2, v3)

    g31 = produto_escalar(v3, v1)
    g32 = produto_escalar(v3, v2)
    g33 = produto_escalar(v3, v3)

    matriz_gram = np.array([
        [g11, g12, g13],
        [g21, g22, g23],
        [g31, g32, g33]
    ])

    # 4. Calcular o determinante da Matriz de Gram (G)
    a, b, c = matriz_gram[0]
    d, e, f = matriz_gram[1]
    g, h, i = matriz_gram[2]

    determinante_gram = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    # O determinante de Gram para vetores linearmente independentes será sempre não-negativo.
    # No entanto, se os vetores forem linearmente dependentes (por exemplo, coplanares),
    # o determinante será zero, resultando em volume zero.
    if determinante_gram < 0:
        if abs(determinante_gram) < 1e-9:
            determinante_gram = 0.0
        else:
            return f"Erro: Determinante de Gram negativo inesperado: {determinante_gram}"


    # 5. Aplicar a fórmula do volume do tetraedro
    volume = (1/6) * np.sqrt(determinante_gram)

    return volume