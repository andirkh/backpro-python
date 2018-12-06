import json
import numpy as np
from index import sigmoid
import sys

with open('./data/campurBig.json') as f:
    data = json.load(f)

weight1 = np.load('./export_weights/weight1.npy')
weight2 = np.load('./export_weights/weight2.npy')

def predict(data, outputTest):
    inputTest = data
    hiddenTest = sigmoid(np.dot(inputTest, weight1))
    output = sigmoid(np.dot(hiddenTest, weight2))
    print(outputTest)
    print(output)

num = int(sys.argv[1])

predict(data[num]["input"], data[num]["output"])
