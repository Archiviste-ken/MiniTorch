from autograd.value import Value
from nn.module import Module


class TestModule(Module):

    def __init__(self):
        self.weight = Value(2)
        self.bias = Value(1)

    def forward(self, x):
        return self.weight * x + self.bias
    

model = TestModule()

print(model.parameters())

model.zero_grad()

print(model.parameters())

