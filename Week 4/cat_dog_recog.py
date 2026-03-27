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
# class network:
#     def __init__:
#         self.layer1=layer()
dataset,info=tfds.load('cats_vs_dogs',with_info=True,as_supervised=True)
print(info.features)
        