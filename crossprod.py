import numpy as np

def crossprod(u, v):
    """
    Calcula o produto vetorial entre dois vetores 3D u e v.
    """
    x = u[1]*v[2] - u[2]*v[1]
    y = u[2]*v[0] - u[0]*v[2]
    z = u[0]*v[1] - u[1]*v[0]
    return np.array([x, y, z])

# Exemplos de vetores
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Produto vetorial
uv = crossprod(u, v)
vu = crossprod(v, u)

print('u x v =', uv)
print('v x u =', vu)

# Produto escalar entre (u x v) e u
dot1 = np.dot(uv, u)
# Produto escalar entre (v x u) e v
dot2 = np.dot(vu, v)

print('(u x v) · u =', dot1)
print('(v x u) · v =', dot2)