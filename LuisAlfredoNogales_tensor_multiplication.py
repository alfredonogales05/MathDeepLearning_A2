#we have implemented the following functions: create_tensor_of_val, calculate_elementwise_product, calculate_matrix_product, calculate_matrix_prod_with_bias, calculate_activation, and calculate_output. These functions allow us to create a tensor of a given value, calculate the elementwise product of two tensors, calculate the matrix product of two tensors, add a bias term to the matrix product, apply an activation function to the result, and calculate the output of a neuron. These functions are essential for building neural networks and performing computations on tensors in PyTorch.
#The create_tensor_of_val function creates a tensor of the given dimensions filled with the specified value.
#The calculate_elementwise_product function calculates the elementwise product of two tensors.
#The calculate_matrix_product function calculates the matrix product of two tensors.
#The calculate_matrix_prod_with_bias function calculates the matrix product of two tensors and adds a bias term.
#The calculate_activation function applies an activation function to the input.
#The calculate_output function calculates the output of a neuron using the functions above.
#These functions can be used to build neural networks and perform computations on tensors in PyTorch.
#The functions are implemented using PyTorch operations and functions, making them efficient and scalable for large datasets and complex models.
#Overall, these functions provide a foundation for building neural networks and performing computations on tensors in PyTorch.


import torch


def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimentions is a tuple of integers.
    Hint: use torch.ones and multiply by val, or use torch.zeros and add val.
    e.g. if dimensions = (2, 3), and val = 3, then the returned tensor should be of shape (2, 3)
    specifically, it should be:
    tensor([[3., 3., 3.], [3., 3., 3.]])
    """
    return torch.ones(dimensions) * val


def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    return A * B


def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i})
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.

    """
    return torch.matmul(X, W.T)


def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i}) and add the bias.
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
       """
    return torch.matmul(X, W.T) + b


def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    Hint: use PyTorch `heaviside` function.
    """
    return torch.heaviside(sum_total, torch.tensor([0.0]))


def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    Hint: use the functions you implemented above.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    return calculate_activation(sum_total)


# Example usage:
# Define the input tensor X, weight tensor W, and bias tensor b
# Note that the dimensions of X and W should be compatible for multiplication.
X = torch.tensor([[1.0, 2.0, 3.0]])
W = torch.tensor([[0.2, 0.4, 0.6]])
b = torch.tensor([0.1])

tensor_val = create_tensor_of_val((2, 3), 3)
print(f"Tensor of val:\n{tensor_val}\n")

elementwise_product = calculate_elementwise_product(X, X)
print(f"Elementwise product:\n{elementwise_product}\n")

matrix_product = calculate_matrix_product(X, W)
print(f"Matrix product:\n{matrix_product}\n")

matrix_product_with_bias = calculate_matrix_prod_with_bias(X, W, b)
print(f"Matrix product with bias:\n{matrix_product_with_bias}\n")

activation = calculate_activation(matrix_product_with_bias)
print(f"Activation:\n{activation}\n")

output = calculate_output(X, W, b)
print(f"Output:\n{output}\n")

