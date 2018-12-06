import json
import numpy as np

# input dataset :
inputDataset = np.empty((0,625), int)
outputDataset = np.empty((0, 3), int)

with open('./data/campur.json') as f:
    data = json.load(f)

with open('./data/campurBig.json') as f2:
    data_test = json.load(f2)

for i in range(len(data)):
    inputDataset = np.append(inputDataset, np.array([data[i]["input"]]), axis=0)
    outputDataset = np.append(outputDataset, np.array([data[i]["output"]]), axis=0)

# params :
alpha= 0.25
epoch = 20000
target_error = 0.0013
neurons = 16
inputLen = 625      #atau inputDataset.shape[1]
outputLen = 3       #atau outputDataset.shape[1]

weight1 = np.random.uniform(-1, 1, (inputLen, neurons))
weight2 = np.random.uniform(-1, 1, (neurons, outputLen))

# activation func :
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    return x * (1 -x)

def relu(x):
    return np.maximum(x,0)

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# cost function
def rms():
    global error_rms
    error_rms = np.sqrt(np.mean(np.square(outputError)))
    return error_rms

def mae():
    global error_mae
    error_mae = np.mean(np.abs(outputError))
    return error_mae
"""
def loss_func():
    global loss
    loss = np.square(outputError)/2
    return loss
"""
# forward
def activate(data):
    global inputLayer, hiddenLayer, outputLayer
    
    inputLayer = data
    hiddenLayer = sigmoid(np.dot(inputLayer, weight1))
    outputLayer = sigmoid(np.dot(hiddenLayer, weight2))

# back
def propagate():
    global outputError, delta2, hiddenError, delta1, chain1, chain2
    #loss_func = np.square(Target - Prediction)/2
    outputError = outputDataset - outputLayer

    dLoss_func = outputError
    delta2 = dLoss_func * dSigmoid(outputLayer)
    chain2 = hiddenLayer.T.dot(delta2)

    hiddenError = delta2.dot(weight2.T)
    delta1 = hiddenError * dSigmoid(hiddenLayer)
    chain1 = inputLayer.T.dot(delta1)
    #print ("Err: " + str(rms()) + "Iter: " + str(i))

def adjust_weight():
    global weight2, weight1
    weight2 += alpha * chain2
    weight1 += alpha * chain1

def predict(data, outputTest):
    inputTest = data
    hiddenTest = sigmoid(np.dot(inputTest, weight1))
    output = sigmoid(np.dot(hiddenTest, weight2))
    print(outputTest)
    print(output)

def trainer():
    #batch
    i = 1
    while i < epoch:
        activate(inputDataset)
        propagate()
        print ("Err: " + str(rms()) + ", Iter: " + str(i))
        adjust_weight()
        if error_rms < target_error:
            break
        i += 1

def save_weights():
    np.save("./export_weights/weight1.npy", weight1)
    np.save("./export_weights/weight2.npy", weight2)



def main():
    # Training :
    trainer()
    # test / predict:
    predict(data_test[4]["input"], data_test[4]["output"])

if __name__ == "__main__":
    main()
