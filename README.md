# Neural Network Playground

This repository shows the implementation of a feed forward neural network library. It is intended for educational purposes and was written to play with (1) the concepts of forward pass and backpropagation and (2) data structures, as discussed a bit in the blog post on this [website](http://pierre.garreau.de).

To start the project, make sure you have [pipenv](https://docs.pipenv.org/) installed. You can then sync and run `main.py`.

```bash
cd app
pipenv sync
pipenv run python main.py
```

Not surprisingly, this toy library sets up a neural network model in a similar fashion as [keras](https://keras.io/). You first define an architecture for your network

```Python
layers = [2, 2, 1]
neural_net = NeuralNet(layers)
```

All activation functions are assumed to be sigmoids. Then you specify your optimizer:

```Python    
optimizer = GradientDescent(options={
    'optimizer': '',
    'maxiter': 1000,
    'tol': 1e-7,
    'jac': True,
    'learning_rate': 1.0
})
```

Finally, you choose the loss function you wish to use:

```Python
loss = Loss.crossentropy
```

One can then fit and predict in a straightforward fashion:
    
```Python
res = neural_net.fit(X_train, y_train, optimizer, loss)
predicted = neural_net.predict(X_test)
for p, y in zip(predicted, y_test):
    print(p,y)
loss = loss(predicted, y_test)
print('Loss: ', loss)
```
