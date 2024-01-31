import os
import matplotlib.pyplot as plt
import random

folderpath = "Pictures3D_TempStorage_Class0/"
structuralSimilarityDistribution = [0 for i in range(10001)]
structuralSimilarityDistribution_SummedUp = []
sum = 0

for file in os.scandir(folderpath):
    structuralSimilarityDistribution[int(str.split(file.name[:-4], "_")[-1])] += 1

for val in structuralSimilarityDistribution:
    sum += val
    structuralSimilarityDistribution_SummedUp.append(sum)

#plt.plot(structuralSimilarityDistribution[9750:])
#plt.show()

for i in range(501):
    print(9500+i, structuralSimilarityDistribution_SummedUp[9500+i])

folderpath_dest = "Pictures3D_Input/0/"

for i, file in enumerate(os.scandir(folderpath)):
    if int(str.split(file.name[:-4], "_")[-1]) <= 9927 or random.random() > 0.9:
        os.replace(file.path, folderpath_dest + file.name)
    if i % 1000 == 0:
        print(i)