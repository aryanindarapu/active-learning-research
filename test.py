import numpy as np

temp = [1, 5, 2, 5, 2, 1, 2,3 ,6 ,8 , 7, 2, 3]

print(np.random.choice(temp, 4, replace=False))

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