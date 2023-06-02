import matplotlib.pyplot as plt
import os

z = []
y = []

file1 = open("C:\Modeling\SEM_6_Modeling\lab_03\out.txt", "r")

while True:
    # считываем строку
    line = file1.readline()
    # прерываем цикл, если строка пустая
    if not line:
        break
    arr = [float(x) for x in line.split()]
    print(arr)
    z.append(arr[0])
    y.append(arr[1])


# закрываем файл
file1.close

plt.xlabel("z = r/R")
plt.ylabel("T(z)")
plt.plot(z, y, label = "График")
plt.axis([0, 1, 0, 2500])
plt.legend()
plt.show()
