import numpy as np

# zadanie1
array = np.array([5]*50)
print(array)

print()
# zadanie2
tablica = np.random.randint(1, 25, (5, 5))
print(tablica)

print()
# zadanie3
tablica = np.arange(10,50+1,2)
print(tablica)

print()
# zadanie4
zera = np.zeros((5, 5), int)
np.fill_diagonal(zera, 8)
print(zera)

print()
# zadanie5
tablica = np.random.randint(1, 5[0.01], (5, 5))
print(tablica)
