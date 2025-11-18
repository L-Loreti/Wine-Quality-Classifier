import numpy as np

# tentativa de leitura das features (TERMINAR DEPOIS)
file = open('[Best Features].txt', 'r')

best_features = np.empty(shape(len(model_list), len(xTrain.columns + 1)))

for i in range(len(model_list)):

    line = file.readline()
    lineList = line.split(', ')
    lineList = [item.strip() for item in lineList]
    while line:
        print(lineList)
        line = file.readline()
        lineList = line.split(', ')
        lineList = [item.strip() for item in lineList]

file.close()
