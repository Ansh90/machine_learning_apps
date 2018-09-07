import numpy as np

a = np.array([(2,3,4),(5,6,7)])
print(a)
b = np.array([(1,1,1),(1,1,1)])
print(b)

c = np.add(a,b)
print("c")
print(c)

al = np.ones(10,)
print(al)
featureVec = np.ones((11,), dtype="float32")
print(featureVec)
featureVec = np.add(al, featureVec)
print("d")
print(d)