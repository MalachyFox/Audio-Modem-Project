import  py.ldpc as ldpc
from py.ldpc_awgn import ch2llr
import numpy as np

c = ldpc.code('802.11n','5/6',54)

from encoder import random_binary

data = random_binary(540*3)

print(data[:10])
x =c.encode(data)

y = x
print(y)
y = (.5-x)*1
y[0] = -0.9
print(y[:10])

#648

yl = ch2llr(y,1)

print(yl[:10])
app, it = c.decode(yl)

print(app[:10])
print(len(app))
print(it)