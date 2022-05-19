import time
import random
import numpy as np

print("\nРезультат работы программы:")
rows = int(input("Введите количество строк (столбцов) квадратной матрицы больше 3 : "))
while rows < 4 :
    rows = int(input("Вы ввели неверное число\nВведите количество строк (столбцов) квадратной матрицы больше 3 :"))
K = int(input("Введите число К="))
A = np.zeros((rows,rows),dtype=int)
F = np.zeros((rows,rows), dtype=int)
t0 = time.time()

for i in range (rows):     #формируем матрицу А
    for j in range(rows):
        #A[i][j]=np.random.randint(-10,10)
        A[i][j] = i*10+j

for i in range (rows):      #формируем матрицу F, копируя из матрицы А
    A[i][j] = 0
F = A.copy()

count1=0
count2=0
for i in range(rows//2,rows):      #считаем кол-во нулей в четных и нечетных столбцах матрицы С
    for j in range(i + 1, rows, 1):
        if j % 2 == 1 and A[i][j] == 0:
            count1 += 1
        elif j % 2 == 0 and A[i][j] == 0:
            count2 += 1

if count1>count2:    # в С кол-во нулевых элементов в нечетных столбцах больше,чем количество нулевых элементов
                    # в четных столбцах, то меняем местами С и В симметрично
    F[0:rows // 2, rows // 2 + rows % 2:rows] = A[rows // 2 + rows % 2:rows, rows // 2 + rows % 2:rows]
    F[rows // 2 + rows % 2:rows, rows // 2 + rows % 2:rows] = A[0:rows // 2, rows // 2 + rows % 2:rows]
else:  #С и Е меняем местами несимметрично
    F[0:rows // 2, 0:rows // 2] = A[rows // 2 + rows % 2:rows, rows // 2 + rows % 2:rows]
    F[rows // 2 + rows % 2:rows, rows // 2 + rows % 2:rows] = A[0:rows // 2, 0:rows // 2]

print("\n A, F time =  \n")
print(A)
print(F)
#
x = []
for i in range(rows):
    for j in range(rows):
        x.append(A[i][j])
c = min(x)
#
if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:
    print("\nМатрица A или F вырождена => нельзя вычислить")
elif np.linalg.det(A) > sum(F.diagonal()):
    A = np.dot(np.dot(A, np.transpose(A)), K * np.transpose(F))  # 1 формула
else:
    print("___G___")
    print(np.tril(A))
    A = (np.transpose(A) + np.tril(A) - np.linalg.inv(F)-1) # 2 формула
print("Rezult\nProgram time: " + str(time.time() - t0) + "seconds.")
print(A)

#for i in A:  # делаем перебор всех строк матрицы
    #for j in i:  # перебираем все элементы в строке
        #print("%5d" % j, end=' ')
    #print()

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

plt.title("Plot")    # 1 пример matplotlib
plt.xlabel("Numbers")
plt.ylabel("volumes")
for j in range (rows):
    plt.plot([i for i in range(rows)], A[j][::], marker='x')
plt.show()

plt.title("Scatter")    # 2 пример matplotlib
plt.xlabel("Numbers")
plt.ylabel("volumes")
for j in range (rows):
    plt.scatter([i for i in range(rows)] , A[j][::])
plt.show()

a = np.diag(range(rows)) # 3 пример matplotlib
plt.matshow(a)
plt.show()

sns.set_theme(style="white")  # 4 пример seaborn
uniform_data = A
if rows >= 50 or K >= 10:
    graph = sns.heatmap(A, vmin=-20 * rows, vmax=20 * rows)
elif rows < 50 or K < 10:
    graph = sns.heatmap(A, vmin=-50, vmax=50, annot_kws={'size': 7}, annot=True, fmt=".1f")
plt.show()

df = pd.DataFrame(A)

p = sns.lineplot(data=df)   # 5 пример seaborn
p.set_xlabel("Номер элемента в столбце", fontsize=10)
p.set_ylabel("Значение", fontsize=10)
plt.show()

sns.heatmap(data = F, annot = True)             # 6 пример seaborn
plt.xlabel("Matrix column number")
plt.ylabel("Matrix row number")
plt.show()