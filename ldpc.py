import numpy as np
from pyldpc import make_ldpc, decode

# Modified from pyldpc package
def binaryproduct(X, Y):

    A = np.dot(X,Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2

# Modified from pyldpc package
def encode(tG, v):

    n, k = tG.shape
    d = binaryproduct(v,tG)

    #x = (-1) ** d

    return d

def ldpc_encode(G,message):
    #H, G = make_ldpc(n, d_v, d_c,systematic=True,seed=1)
    tG = np.transpose(G)
    encoded_message = encode(tG, message)

    return encoded_message

def ldpc_decode(H, encoded_message, SNR=100):
    #H, G = make_ldpc(n, d_v, d_c)
    decoded_message = decode(H, encoded_message, SNR)
    
    return decoded_message