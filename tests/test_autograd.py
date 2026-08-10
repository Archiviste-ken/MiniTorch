from autograd.value import Value


def test_addition():
    a = Value(2)
    b = Value(3)

    c = a + b
    c.backward()

    assert c.data == 5
    assert a.grad == 1
    assert b.grad == 1
    
def test_multiplication():
    a = Value(2)
    b = Value(3)

    c = a * b
    c.backward()

    assert c.data == 6
    assert a.grad == 3
    assert b.grad == 2
    
def test_multiple_paths():
    a = Value(2)
    b = Value(3)

    c = a * b
    d = c + a

    d.backward()

    assert a.grad == 4
    assert b.grad == 2
    
def numerical_gradient(f, x, h=1e-6):
    x_plus = x + h
    x_minus = x - h

    return (f(x_plus) - f(x_minus)) / (2 * h)