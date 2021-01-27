#Activation functions

def sigmoid(x):
    """
    Sigmoid function and data for backpropagation
    """
    A = 1/(1+np.exp(-x))
    temp = x
    return A, temp

def tanh(x):
    """
    tanh activation and data for backpropagation
    """
    A = np.tanh(x)
    temp = x
    return A, temp

def relu(x):
    """
    ReLU activation function and data for backpropagation
    """
    A = np.maximum(0,x)  
    temp = x
    return A, temp

def leaky_relu(x):
    """
    Leaky ReLU activation function and data for backpropagation
    """
    A = np.maximum(0.01*x,x)  
    temp = x
    return A, temp