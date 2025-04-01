import numpy as np

print("Este programa calcula a posição do efetuador final de um robô planar de dois elos.")
theta1_deg = float(input("Digite o ângulo θ1 de deslocamento: "))
theta2_deg = float(input("Digite o ângulo θ2 de deslocamento: "))

def posicao_efetuador(theta1_deg, theta2_deg):
    L1 = 20.0
    L2 = 15.0
    
    # Converter para radianos
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)
    
    # Calcular posição
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    
    # Arredondar para 1 casa decimal
    return round(x, 1), round(y, 1)

print("O ponto final é: ",np.array(posicao_efetuador(theta1_deg, theta2_deg)))