#Forward Difference Method
#The forward difference method is a simple way to approximate the derivative of a function.
#It is based on the definition of the derivative:
#f'(x) = lim(h->0) (f(x + h) - f(x)) / h
#In the forward difference method, I choose a small value of h and calculate the derivative as:

def derive(f, x, h=0.0001):
    return (f(x + h) - f(x)) / h

# Example usage:
def example_function(x):
    return x**2

x = 2
print(f"The derivative of x^2 at x={x} is approximately {derive(example_function, x)}")
# The derivative of x^2 at x=2 is approximately 4.000199999999799


#Central Difference Method
#The central difference method is another simple way to approximate the derivative of a function.
#It is based on the definition of the derivative:
#f'(x) = lim(h->0) (f(x + h) - f(x - h)) / (2 * h)
#In the central difference method, I choose a small value of h and calculate the derivative as:

def derive(f, x, h=0.0001):
    return (f(x + h) - f(x - h)) / (2 * h)

# Example usage:
def example_function(x):
    return x**2

x = 2
print(f"The derivative of x^2 at x={x} is approximately {derive(example_function, x)}")
# The derivative of x^2 at x=2 is approximately 4.000000000000001

