import numpy as np
from tensorflow.keras.datasets import mnist

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # Khởi tạo trọng số nhỏ để tránh bão hòa hàm Sigmoid
        self.weight = np.random.randn(n_inputs, n_neurons) * 0.01
        self.bias = np.zeros((1, n_neurons))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weight) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, gradient_from_next_layer, learning_rate=0.1):
        # gradient_from_next_layer chính là "y" (tín hiệu lỗi) truyền vào cho lớp này
        # dZ = error * sigmoid_derivative
        dZ = gradient_from_next_layer * self.output * (1 - self.output)
        
        # Tính đạo hàm để cập nhật tham số lớp hiện tại
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        
        # Tính "y" (lỗi) để truyền ngược về cho lớp phía trước
        # Công thức: error_prior = dZ * W.T
        dX = np.dot(dZ, self.weight.T)

        # Cập nhật trọng số và bias
        self.weight -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return dX

class NeuralNetwork:
    def __init__(self):
        # 784 (input) -> 128 -> 64 -> 10 (output)
        self.layer1 = Layer(784, 128)
        self.layer2 = Layer(128, 64)
        self.layer3 = Layer(64, 10)

    def fit(self, X, y, epochs=10, lr=0.1):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # --- 1. Forward Pass ---
            out1 = self.layer1.forward(X)
            out2 = self.layer2.forward(out1)
            out3 = self.layer3.forward(out2)
            error_output = (out3 - y) / n_samples
            grad2 = self.layer3.backward(error_output, lr)
            grad1 = self.layer2.backward(grad2, lr)
            self.layer1.backward(grad1, lr)
            loss = np.mean(np.square(out3 - y))
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

    def predict(self, X):
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        return self.layer3.forward(out2)

    def calculate_accuracy(self, X, y_true_oh):
        output = self.predict(X)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(y_true_oh, axis=1)
        return np.mean(predictions == labels)


# --- Chuẩn bị dữ liệu ---
print("Dang tai du lieu MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0
# Chuyển nhãn sang dạng One-hot encoding (10 cột)
y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

# --- Huấn luyện mô hình ---
model = NeuralNetwork()
model.fit(X_train, y_train_oh, epochs=20, lr=0.001)

# --- Kiểm tra độ chính xác ---
acc = model.calculate_accuracy(X_test, y_test_oh)
print(f"\nDo chinh xac tren tap kiem tra: {acc * 100:.2f}%")