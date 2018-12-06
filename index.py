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
epoch = 20000
target_error = 0.0013
neurons = 20
inputLen = 625      #atau inputDataset.shape[1]
outputLen = 3       #atau outputDataset.shape[1]
weight1 = 2 * np.random.random((inputLen, neurons)) - 1 
weight2 = 2 * np.random.random((neurons, outputLen)) - 1

# activation func :
def sigmoid(x):
    return 1/(1+np.exp(-x))

def grad_sigmoid(x):
    return x * (1 -x)

def rms():
    global error_rms
    error_rms = np.sqrt(np.mean(np.square(outputError)))
    return error_rms 

def feed_forward(data):
    global inputLayer, hiddenLayer, outputLayer
    
    inputLayer = data
    hiddenLayer = sigmoid(np.dot(inputLayer, weight1))
    outputLayer = sigmoid(np.dot(hiddenLayer, weight2))

def propagate():
    global outputError, delta2, hiddenError, delta1
    outputError = outputDataset - outputLayer
    delta2 = outputError * grad_sigmoid(outputLayer)
    hiddenError = delta2.dot(weight2.T)
    delta1 = hiddenError * grad_sigmoid(hiddenLayer)
    #print ("Err: " + str(rms()) + "Iter: " + str(i))

def adjust_weight():
    global weight2, weight1
    weight2 += hiddenLayer.T.dot(delta2)
    weight1 += inputLayer.T.dot(delta1)

def predict(data, outputTest):
    inputTest = data
    hiddenTest = sigmoid(np.dot(inputTest, weight1))
    output = sigmoid(np.dot(hiddenTest, weight2))
    print(outputTest)
    print(output)

def trainer():
    i = 1
    while i < epoch:
        feed_forward(inputDataset)
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
    predict(data_test[2]["input"], data_test[2]["output"])

if __name__ == "__main__":
    main()
