
import time

combination = []

def subset_sum(numbers, target, partial=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target and partial not in combination:
        combination.append(partial)

    # if we reach the number why bother to continue
    if s >= target:
        return

    if partial in combination:
        return

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i+1:]
        subset_sum(remaining, target, partial + [n])

    return combination


user = 20
tgt = user * 2
l1 = [8 for i in range(round(tgt/8))]
l2 = [6 for i in range(round(tgt/6))]
l3 = [4 for i in range(round(tgt/4))]
l4 = [2 for i in range(round(tgt/2))]

file = 'Bit/user{}_bit{}.csv'.format(user, tgt)

l = [str(i) for i in subset_sum(l1+l2+l3+l4,tgt)]

with open(file, mode='w') as f:
    f.write('\n'.join(l))


