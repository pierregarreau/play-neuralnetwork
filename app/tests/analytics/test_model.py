
def test_fit():
    layers = [2, 1]
    neural_net = NeuralNet(layers)
    features, labels = data.load.dummy()
    optimizer = GradientDescent(options={'optimizer': '', 'maxiter': 1000, 'tol': 1e-7})
    loss = analytics.util.logErrorClassificationFunction
    res = neural_net.fit(features, labels, optimizer, loss)
    print(res)
    assert loss(predicted, true) < 1e-1
