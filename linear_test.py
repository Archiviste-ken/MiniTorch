from autograd.value import Value
from nn.linear import Linear

layer = Linear(2, 3)

x = [
    Value(2.0),
    Value(3.0)
]

output = layer(x)

print(len(output))

loss = output[0] + output[1] + output[2]

loss.backward()

for parameter in layer.parameters():
    print(parameter.grad)