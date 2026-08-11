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
    


# from autograd.value import Value
# from nn.linear import Linear


# # Create a Linear layer
# # 3 inputs → 2 neurons → 2 outputs
# layer = Linear(3, 2)


# # Create input values
# inputs = [
#     Value(2.0),
#     Value(3.0),
#     Value(4.0)
# ]


# # Run the layer
# outputs = layer(inputs)


# # Print results
# print("Inputs:")
# print(inputs)

# print("\nOutputs:")
# print(outputs)