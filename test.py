import numpy as np
from collections import Counter

n_train_views = []
for i in range(4):
    n_train_views.extend(50 * [i])

np.random.shuffle(n_train_views)
c = Counter(n_train_views[:160])

for idx, val in enumerate([1, 2, 3, 4]):
    print(c[idx])

quit()
test = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1]])

print(test)
for i in range(len(test)):
    for j in range(len(test[0])):
        if test[i, j] == 1:
            topBorder, botBorder = 0 if i-1 < 0 else i-1, len(test) if i+1 > len(test)-1 else i+2
            leftBorder, rightBorder = 0 if j-1 < 0 else j-1, len(test[0]) if j+1 > len(test[0])-1 else j+2
            print(test[topBorder:botBorder, leftBorder:rightBorder])
            
print(list(zip(*test.nonzero())))
# print(test[test == 1])