import numpy as np

print()
# zadanie1
array = np.array([5]*50)
print(array)

print()
# zadanie2
tablica_zad2 = np.arange(1,26).reshape(5, 5)
print(tablica_zad2)

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
tablica = np.arange(0.01,1.01,0.01)
print(tablica.reshape((10,10)))

print()
# zadanie6
print(np.linspace(0,1,50))

print()
# zadanie7
podtablica = tablica_zad2[2:5,1:5]
print(podtablica)

print()
# zadanie8
podtablica = tablica_zad2[:3,4].reshape(3,1)
print(podtablica)

print()
# zadanie9
suma= tablica_zad2[3]+tablica_zad2[4]
print(suma.sum())

print()
# zadanie10
def random_tensor():
    tmp = np.random.randint(1,10,(1,5))
    return np.random.randint(1,50,(tmp[0][0], tmp[0][1]))
print(random_tensor())