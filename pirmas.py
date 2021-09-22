from typing import Sized


print("Hello!")
a = 4
b = a * 3;
print(b)

for i in range(0, 10):
  e =a*i
  print(e)
  print("Tekstas: {}".format(e))

print("------------------")
arr1 = list()
arr2 = []
arr2 = [1, 2, 3, 4]
print(arr2)
arr4 = [1, 2.5, "dfd"]
print(arr4)
print(len(arr4))
print("------------------")

dydis = len(arr4)
for i in range(0, dydis):
  print(arr4[i])
  if 2 == i:
      arr4[i] = 44
print(arr4)