
import csv
import numpy as np
user = 20
total_bit = user *2
SNR = 20

inp = "Data/train_data/channel/user{}_bit{}_SNR={}.csv".format(user, total_bit, SNR)
ans = "Data/train_data/bit/user{}_bit{}_SNR={}.csv".format(user, total_bit, SNR)
ans_dict = "Data/train_data/bit_dict/user{}_bit{}.csv".format(user, total_bit)

input = []
a_dict = []
answer = []

with open(inp, 'r') as f:
    for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
        input.append(row)

with open(ans_dict, 'r') as f:
    for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
        row = [int(i) for i in row]
        a_dict.append(row)

unit = np.eye(len(a_dict), dtype=np.int)

with open(ans, 'r') as f:
    for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
        row = [int(i) for i in row]
        print(row)
        print(unit[a_dict.index(row)])
        answer.append(unit[a_dict.index(row)])

a = list(map(list, set(map(tuple, answer))))

# file =  "Data/train_data/bit_dict/user{}_bit{}.csv".format(user, total_bit)
# with open(file, mode='w') as f:
#     writer = csv.writer(f)
#     for i in a:
#         writer.writerow(i)


tai = np.eye(len(a))
aaan = []
k1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k3 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k4 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k10 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k12 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k13 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

aaan.append(tai[a.index(k1)])
aaan.append(tai[a.index(k2)])
aaan.append(tai[a.index(k3)])
aaan.append(tai[a.index(k4)])

print(np.array(aaan))


c = []
for i in [8,6,4]:
    tmp = round(total_bit/i)
    if i == 8:
        c.append(tmp)
    else:
        c.append(tmp + c[-1])

for i in a:
    b = []
    for j in range(len(i)):
        if j < c[0]:
            if i[j] == 1:
                b.append(8)
        if c[0] <= j < c[1]:
            if i[j] == 1:
                b.append(6)
        if c[1] <= j < c[2]:
            if i[j] == 1:
                b.append(4)
        if c[2] <= j:
            if i[j] == 1:
                b.append(2)

