import numpy as np
from sklearn import datasets
import tensorflow_datasets as tfds
digits=datasets.load_digits()
class layer:
    def __init__(self,n_inputs,n_neurons):
        self.weight=np.random.randn(n_inputs,n_neurons)
        self.bias=np.random.randn(1,n_neurons)
    def forward(self,inputdata):
        self.input=inputdata
        self.z=np.dot(inputdata,self.weight)+self.bias
        self.output=(1/(1+np.exp(self.z)))#tính sigmoid
    def backward(self,gradient_layer,lr):
        delta=(gradient_layer)*self.output*(1-self.output)
        db=np.sum(delta,axis=0,keepdim=True)
        dw=np.dot(self.input.T,delta)
        dx=np.dot(delta,self.weight.T)
        self.weight=self.weight+dw*lr
        self.bias=self.bias+db*lr
        return dx
class network:
    def __init__(self):
        self.layer1=layer()
        self.layer2=layer(128,64)
        self.layer3=layer(64,2)
    def fit(self,intput,y,epochs,lr):
        n_samples = input.shape[0]
        for n in range(epochs):
            o1=self.layer1.forward(input)
            o2=self.layer2.forward(o1)
            self.output=self.layer3.forward(o2)
            self.output=(self.output-y)/n_samples
            dx3=self.layer3.backward(self.output,lr)
            dx2=self.layer2.backward(dx3,lr)
            dx1=self.layer1.backward(dx2,lr)d
            loss=np.mean(np.square(y-self.output))
            print(loss)
            print(f"Epoch {n+1}/{epochs} - Loss: {loss:.6f}")
    def predict(self, X):
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        self.tOutput= self.layer3.forward(out2)

    def calculate_accuracy(self, X, y_true_oh):
        self.predict(X)
        predictions = np.argmax(self.tOutput, axis=1)
        labels = np.argmax(y_true_oh, axis=1)
        return np.mean(predictions == labels)
    
dataset,info=tfds.load('cats_vs_dogs',with_info=True,as_supervised=True)
print(info.features)
        