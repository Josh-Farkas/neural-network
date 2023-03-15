import numpy as np


def cost_derivative(output, correct_output):
    return output - correct_output



# the sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    # return np.maximum(np.zeros(z.shape), z)
    return z * (z > 0)

def ReLU_prime(z):
    # return np.where(z > 0, 1, 0)
    return 1 * (z > 0)

def linear(z):
    return z

def linear_prime(z):
    return 1


def correlate(img, kernel, padding=0, strides=1):
    """
    Takes a filter matrix and takes the dot product with the input image as
    it moves across the whole image, producing an output matrix
    """

    img_x = img.shape[1]
    img_y = img.shape[0]
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]
    output_x = int((img_x + 2*padding - kernel_x) / strides + 1)
    output_y = int((img_y + 2*padding - kernel_y) / strides + 1)

    output = np.zeros((output_y, output_x), dtype=float)

    if padding != 0:
        padded_img = np.zeros((img_y + padding*2, img_x + padding*2))
        padded_img[padding:-padding, padding:-padding] = img
    else:
        padded_img = img

    for x in range(output_x):
        for y in range(output_y):
            output[y, x] = (padded_img[y:y+kernel_y, x:x+kernel_x] * kernel).sum()
    
    return output


def convolve(img, kernel, padding=0, strides=1):
    """
    Cross correlate with flipped kernel
    """
    return correlate(img, np.fliplr(np.flipud(kernel)), padding=padding, strides=strides)




funcs = {
    "relu":(ReLU, ReLU_prime),
    "sigmoid":(sigmoid, sigmoid_prime),
    "linear": (linear, linear_prime)
}