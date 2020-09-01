
import numpy as np

from Model.Convcode import Trellis, conv_encode, viterbi_decode, puncturing, depuncturing

symbol = 100
array = np.random.randint(0, 2, symbol)

rate = 2/3
code_symbol = int(symbol * 1/rate)
L = 7  # constraint length
memory = np.array(L-1, ndmin=1) # number of delay elements
g_matrix = np.array((0o171, 0o133), ndmin=2)
tb_depth = 5*(memory.sum() + 1)
trellis = Trellis(memory, g_matrix)

if rate == 1/2:
    punct_vec = [1,1]
    puncture_matrix = np.array([[1],[1]])
elif rate == 2/3:
    punct_vec = [1,1,0,1]
    puncture_matrix = np.array([[1,1,0,1]], dtype=np.float)
elif rate == 3/4:
    punct_vec = [1,1,1,0,0,1]
    puncture_matrix = np.array([[1,0,1],[1,1,0]])
elif rate == 5/6:
    punct_vec = [1,1,1,0,1,0,1,0]
    puncture_matrix = np.array([[1,0,1,0,1],[1,1,0,1,0]])
elif rate == 7/8:
    punct_vec = [1,1,1,0,0,1,1,0,0,1]
    puncture_matrix = np.array([[1,0,0,0,1,0,1],[1,1,1,1,0,1,0]])

encode = conv_encode(array, trellis, puncture_matrix=puncture_matrix)
encode = encode[:symbol*2]
shouldbe = len(encode)
punc = puncturing(encode, punct_vec)
print(len(punc))
depunc = depuncturing(punc, punct_vec, shouldbe)
decode = viterbi_decode(depunc.astype(float), trellis, tb_depth)

print(array)
print(encode)
print(decode[:len(array)])
cnt = 0
for i in range(len(array)):
    if array[i] != decode[:len(array)][i]:
        cnt += 1
print(cnt)

