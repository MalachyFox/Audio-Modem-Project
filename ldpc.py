import numpy as np
from pyldpc import make_ldpc, decode

# Modified from pyldpc package
def binaryproduct(X, Y):
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2

# Modified from pyldpc package
def encode(tG, v):
    n, k = tG.shape
    d = binaryproduct(tG, v)
    x = (-1) ** d

    return x

def ldpc_encode(n, d_v, d_c, message, SNR):
    H, G = make_ldpc(n, d_v, d_c)
    encoded_message = encode(G, message, SNR)

    return encoded_message

def ldpc_decode(n, d_v, d_c, encoded_message, SNR):
    H, G = make_ldpc(n, d_v, d_c)
    decoded_message = decode(H, encoded_message, SNR)
    
    return decoded_message