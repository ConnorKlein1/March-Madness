import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

class Layer():
    def __init__(self, *shape):
        self.m_weights = np.zeros((shape[1], shape[0]), dtype=np.float64)
        self.m_deltaWeights = np.zeros((shape[1], shape[0]), dtype=np.float64)
        self.m_biasWeight = np.zeros((shape[1],1), dtype=np.float64)
    
    def InitializeWeights(self, low, high):
        self.m_weights = np.random.uniform(low, high, self.m_weights.shape).astype(np.float64)
        self.m_biasWeight = np.random.uniform(low, high, self.m_biasWeight.shape).astype(np.float64)
    
    # Only works when m_weights is 1d
    def Save(self, dataDictionary, name):
        dataDictionary[f'Layer{name}_Weights'] = self.m_weights
        dataDictionary[f'Layer{name}_DeltaWeights'] = self.m_deltaWeights
        dataDictionary[f'Layer{name}_Biases'] = self.m_biasWeight
    
    def Load(self, dataDictionary, name):
        self.m_weights = dataDictionary[f'Layer{name}_Weights']
        self.m_biasWeight = dataDictionary[f'Layer{name}_Biases']
        self.m_deltaWeights = dataDictionary[f'Layer{name}_DeltaWeights']

class Sigmoid(Layer):
    def __init__(self, *shape):
        super().__init__(*shape)
        
    def F(self, x):
        with np.errstate(over='raise'):
            x = np.maximum(x, -700)
            y = 1 / (1 + np.exp(-x))
            return y
    
    def dF(self, x):
        return self.F(x) * (1 - self.F(x))

    def Forward(self, x):
        z = np.matmul(self.m_weights, x) + self.m_biasWeight
        return self.F(z)
    
    def Backward(self, a, x, Y, delta, piorWeights, momentum, learningRate, lossFunction):
        # Output Layer
        if delta is None:
            # dJ/dw_ij = f'(s_i)[y_i-y_hat_i] * h_j
            delta = self.dF(a) * lossFunction(Y, a)
            loss =  np.matmul(delta, x.T)
        # Hidden Layer
        else:
            # dJ/dw_jk = f'(s_j)S<W_ij * delta_i * x_k>
            delta = self.dF(a) * np.matmul(piorWeights.T, delta)
            loss = np.matmul(delta, x.T) 
    
        dW = learningRate * loss
        
        self.m_weights += dW + momentum * self.m_deltaWeights
        self.m_biasWeight += learningRate * delta
        
        self.m_deltaWeights = dW
    
        return delta
    
class WinnerTakeAll(Layer):
    def __init__(self, *shape):
        self.m_weights = np.random.rand(*shape)
        
    def Forward(self, x):
        # Convert x back to old form (row vector)
        x = x.reshape(x.shape[0],)
        
        norm = np.linalg.norm((self.m_weights - x), axis=2)
        winningNueron = np.argmin(norm)
        featureSpace1D = np.zeros((self.m_weights.shape[0] * self.m_weights.shape[1], 1))
        featureSpace1D[winningNueron] = 1
        return featureSpace1D
    
    def Save(self, dataDictionary, name):
        dataDictionary[f'm_featureSpace'] = self.m_weights
        
    def Load(self, dataDictionary, name):
        self.m_weights = dataDictionary[f'm_featureSpace']
    
    
class NeuralNetwork():
    def __init__(self, composition, lossFunction = None, errorFunction = None):
        self.m_net = composition
        self.m_learningRate = 10 ** -5
        self.m_momentum = 0.9
        self.m_currentError = 10 ** 20
        self.m_leastError = 10 ** 20
        self.m_errorHistory = []
        self.m_errorHistoryTest = []
        self.m_forawrdStack = deque()
        self.m_h = 0.75
        self.m_l = 0.25
        
        if lossFunction is not None:
            self.m_lossFunction = lossFunction
        else:
            self.m_lossFunction = self.Loss
            
        if errorFunction == "Regression":
            self.m_errorFunction = self.Regression
        elif errorFunction is not None:
            self.m_errorFunction = errorFunction
        else:
            self.m_errorFunction = self.CalculateError
        
    def InitializeRandomWeights(self, low:int, high:int):
        for layer in self.m_net:
            layer.InitializeWeights(low, high)
            
    def InitializeWeights(self, fileList):
        for i, layer in enumerate(self.m_net):
            if fileList[i] == 'random':
                layer.InitializeWeights(0, 0.5)
            else:
                data = np.load(fileList[i], allow_pickle=True).item()
                layer.Load(data, i)
            
    def Forward(self, X):
        self.m_forawrdStack.clear()
        self.m_forawrdStack.append(X)
        for layer in self.m_net:
            self.m_forawrdStack.append(layer.Forward(self.m_forawrdStack[-1]))
        
    def Backwards(self, Y, staticLayer = None):
        delta = None
        
        for i, layer in reversed(list(enumerate(self.m_net))):
            # If layer can't be trained, can't train the rest of them either
            if i == staticLayer:
                break
            
            a = self.m_forawrdStack.pop()
            x = self.m_forawrdStack[-1]
            
            if delta is None:
                delta = layer.Backward(a, x, Y, delta, None,
                           self.m_momentum, self.m_learningRate, self.m_lossFunction)
            else:
                delta = layer.Backward(a, x, Y, delta, self.m_net[i+1].m_weights,
                           self.m_momentum, self.m_learningRate, None)
        
    def Train(self, X, epochs, Y=None, X_test=None, Y_test=None, fileName='NNSave', staticLayer = None):
        if Y is None:
            Y = deepcopy(X)
        if X_test is None:
            X_test = deepcopy(X)
        if Y_test is None:
            Y_test = deepcopy(X)
            
        for epoch in tqdm(range(epochs), desc="Training"):
            permuation = np.random.permutation(X.shape[0])

            for i in range(X.shape[0]):
                # Need to reshape this into a (n,1) from (n,), why? bc np
                x = X[permuation[i]].reshape(X[permuation[i]].shape[0],1)
                y = Y[permuation[i]].reshape(Y[permuation[i]].shape[0],1)
                self.Forward(x)
                self.Backwards(y, staticLayer)
            
            if epoch % 1  == 0:
                self.m_errorFunction(X, Y, X_test, Y_test)
                
                if self.m_currentError < self.m_leastError:
                    self.m_leastError = self.m_currentError
                    self.Save(f'{fileName}.npy')
                
        self.Save(f'{fileName}_End.npy')
            
    def Loss(self, Y, Y_hat):
        return Y - Y_hat
    
    def BoundedLoss(self, Y, Y_hat):
        Y_hat = np.where(Y_hat >= self.m_h, 1, np.where(Y_hat <= self.m_l, 0, Y_hat))
        return self.Loss(Y, Y_hat)
    
    def Predict(self, x):
        for layer in self.m_net:
            x = layer.Forward(x)

        return x
    
    def j2Loss(self, X, Y=None):
        if Y is None:
            Y = deepcopy(X)
        
        X = self.Predict(X)

        loss = np.square(X-Y)
        loss = 0.5*np.sum(loss)
        return loss
    
    # Keep arguement Y so we can satisfy the errorFunction condidiiton
    def Regression(self, X, Y, X2, Y2):
        error = 0
        for i in range(X.shape[0]):
            x = X[i].reshape(X[i].shape[0],1)
            y = Y[i].reshape(Y[i].shape[0],1)
            error += self.j2Loss(x, y)
            
        self.m_currentError = error
        self.m_errorHistory.append(error)
        
        error = 0
        for i in range(X2.shape[0]):
            x = X2[i].reshape(X2[i].shape[0],1)
            y = Y2[i].reshape(Y2[i].shape[0],1)
            error += self.j2Loss(x, y)
        
        self.m_errorHistoryTest.append(error)
        
    def CalculateError(self, train_data, train_labels, test_data, test_labels):
        positives = 0
        for i in range(train_labels.shape[0]):
            x = train_data[i].reshape(train_data[i].shape[0],1)
            y = train_labels[i].reshape(train_labels[i].shape[0],1)
            prediction = int(np.round(self.Predict(x)[0], 0).item())
            real = int(np.round(y,0).item())
            
            if prediction == real:
                positives += 1
                
        self.m_currentError = 1 - (positives / train_labels.shape[0])
        self.m_errorHistory.append(1 - (positives / train_labels.shape[0]))
        
        positives = 0
        for i in range(test_labels.shape[0]):
            x = test_data[i].reshape(test_data[i].shape[0],1)
            y = test_labels[i].reshape(test_labels[i].shape[0],1)
            prediction = int(np.round(self.Predict(x)[0], 0).item())
            real = int(np.round(y,0).item())
            
            if prediction == real:
                positives += 1
                
        self.m_errorHistoryTest.append(1 - (positives / test_labels.shape[0]))

    def Save(self, fileName):
        with open(fileName, 'wb') as f:     
            data = {}
            for i, layer in enumerate(self.m_net):
                layer.Save(data, i)
            
            data['ErrorHistory'] = self.m_errorHistory
            data['ErrorHistoryTest'] = self.m_errorHistoryTest
            data['CurrentError'] = self.m_currentError
            data['LeastError'] = self.m_leastError
            np.save(f, data)
                
    def Load(self, fileName):
        data = np.load(fileName, allow_pickle=True).item()
        self.m_errorHistory = data['ErrorHistory']
        self.m_errorHistoryTest = data['ErrorHistoryTest']
        self.m_currentError = data['CurrentError']
        self.m_leastError = data['LeastError']
        
        for i, layer in enumerate(self.m_net):
            layer.Load(data, i)
    
    ### PLOTTING ###
    
    def PlotErrorHistory(self):
        x = [10*i for i in range(len(self.m_errorHistory))]
        plt.plot(x, self.m_errorHistory, label='Training')
        plt.plot(x, self.m_errorHistoryTest, label='Test')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Error %")
        plt.title("Error History")
        plt.show()
        
    def PlotHiddenLayer(self, layerNumber, rows, cols):
        # Randomly select rows and cols
        # randomArray = np.arange(0, self.m_net.m_weights.shape[0])
        permuation = np.random.permutation(self.m_net[layerNumber].m_weights.shape[0])
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
        
        counter = 0
        for row in range(rows):
            for col in range(cols):
                w = self.m_net[layerNumber].m_weights[permuation[counter]]
                
                weightsSquare = np.sqrt(w.size)
                
                # Normalize Weights
                normalizedW = w / np.max(w)
                
                if not weightsSquare.is_integer():
                    return ValueError(f"{weightsSquare} is not integer")
                
                # convert n*n,1 array to n,n array
                weightsSquare = int(weightsSquare)
                reshapedW = normalizedW.reshape((weightsSquare, weightsSquare))
        
                axes[row][col].imshow(reshapedW)
                counter += 1
                
        fig.suptitle("Hidden Layer Images")
        plt.tight_layout()
        plt.show() 
    
    def PlotLoss(self, trainingSet, testSet):
        trainDigitSize = int(trainingSet.shape[0] / 10)
        testDigitSize = int(testSet.shape[0] / 10)
        
        # Width of the bars (optional)
        bar_width = 0.2
        # Create a figure and axis
        fig, ax = plt.subplots()
        trainList = []
        testList = []
        catagories = ['0', '1', '2','3','4','5','6','7','8','9']
        
        for i in range(10):
            trainSum = 0
            testSum = 0
            for j in range(i*10, i*10 + trainDigitSize):
                x = trainingSet[j].resize(trainingSet[j].shape[0],1)
                trainSum += self.j2Loss(x)
                
            for j in range(i*10, i*10 + testDigitSize):
                x = testSet[j].resize(testSet[j].shape[0],1)
                testSum += self.j2Loss(x)

            trainSum /= trainDigitSize
            testSum /= testDigitSize
            trainList.append(trainSum)
            testList.append(testSum)
            
        # Calculate the x-coordinates for each set of bars
        x = np.arange((len(catagories)))

        # Create bars for each set of data
        ax.bar(x - bar_width/2, trainList, bar_width, label='Training Loss')
        ax.bar(x + bar_width/2, testList, bar_width, label='Testing Loss')

        # Set the x-axis labels
        # ax.set_xticklabels(catagories)

        # Set axis labels and title
        ax.set_xlabel('Classification')
        ax.set_ylabel('Averge Loss')
        ax.set_title('Avergae Loss on Training and Test Data')
        ax.legend()
        # Show the plot
        plt.show()
        
    def CompareInputOutPut(self, dataSet, numberOfComp):
        shuffled = np.random.permutation(dataSet.shape[0])
        fig, axes = plt.subplots(2, numberOfComp, figsize=(10, 5))

        counter = 0
        for col in range(numberOfComp):
            X = dataSet[shuffled[counter]]
            x = X.reshape(X.shape[0],1)
            
            Y = deepcopy(X)
            X = self.Predict(x)
            
            weightsSquare = np.sqrt(X.size)
            
            # Normalize Weights
            normalizedX = X #/ np.max(X)
            
            # convert n*n,1 array to n,n array
            weightsSquare = int(weightsSquare)
            reshapedX = normalizedX.reshape((weightsSquare, weightsSquare))
            reshapedY = Y.reshape((weightsSquare, weightsSquare))
    
            axes[0][col].imshow(reshapedY)
            axes[1][col].imshow(reshapedX)
            
            counter += 1
                
        fig.suptitle("Input vs Output Image")
        plt.tight_layout()
        plt.show()
        
    def ConfMatrix(self, test_set_data, test_set_labels):
        right = 0
        confusionMatrix = np.zeros((2,2))
        for i in range(test_set_labels.shape[0]):
            x = test_set_data[i].reshape(test_set_data[i].shape[0],1)
            y = test_set_labels[i].reshape(test_set_labels[i].shape[0],1)
            prediction = int(np.round(self.Predict(x)[0], 0).item())
            real = y.item()
            
            confusionMatrix[prediction][real] += 1

            if prediction == real:
                right += 1

        df_cm = pd.DataFrame(confusionMatrix, index = [i for i in "01"],
                        columns = [i for i in "01"])

        plt.imshow(df_cm)
        plt.ylabel('Assigned Class')
        plt.title("Actual Class")
        plt.grid(True, linestyle='--', linewidth=0.5, color='black')
        plt.colorbar()

        for i in range(confusionMatrix.shape[0]):
            for j in range(confusionMatrix.shape[1]):
                plt.annotate(str(int(confusionMatrix[i, j])), (j, i), color='w',
                            fontsize=12, ha='center', va='center')

        plt.show()