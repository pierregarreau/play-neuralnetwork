import pytest
from analytics.activation import ActivationFactory


def test_factory_should_throw_error():
    with pytest.raises(ValueError) as value_error:
        activation_function = ActivationFactory.create('incorrect_type')
    assert str(value_error.value) == 'bad activation function type'


def test_factory_should_return_sigmoid():
    activation_function = ActivationFactory.create('sigmoid')
    assert activation_function.function(0.0) == 0.5
    assert activation_function.derivative(0.0) == 0.25
