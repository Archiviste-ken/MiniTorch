def numerical_gradient(f, x, h=1e-6):
    x_plus = x + h
    x_minus = x - h

    return (f(x_plus) - f(x_minus)) / (2 * h)