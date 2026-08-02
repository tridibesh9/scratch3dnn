import numpy as np
from scratch3dnn import *

model = NeuralNet(
    Layer(2, 4),
    Relu(),
    Layer(4, 1),
    Sigmoid()
)

loss = MSELoss()
optimizer = SGDOptimizer(model.get_params(), learning_rate=0.1)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],[1],[1],[0]])

for epoch in range(10000):
    pred = model.forward(X)

    l = loss.forward(pred, y)

    grad = loss.backward()

    model.backward(grad)

    optimizer.step()
    optimizer.zero_grad()

print(model.forward(X))