import numpy as np


def cost_derivative(x, y):
    return x - y

def squared_error(x, y):
    return (x - y)**2

# mean squared error
def mse(x, y):
    return ((x - y)**2).mean()


# the sigmoid function
def sigmoid(z):
    x = 1.0/(1.0 + np.exp(-z))
    return x

# derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    return z * (z > 0)

def ReLU_prime(z):
    return 1 * (z > 0)

def linear(z):
    return z

def linear_prime(z):
    return 1


def correlate(img, kernel, padding=0, strides=1, dtype=np.float32):
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

    output = np.zeros((output_y, output_x), dtype=dtype)

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
    # Cross correlate with flipped kernel
    return correlate(img, np.fliplr(np.flipud(kernel)), padding=padding, strides=strides)


def maxpool(img, pool_size):
    img_x = img.shape[0]
    img_y = img.shape[1]

    # list slicing will be out of range for non factor values
    output_x = int(np.ceil(img_x / pool_size))
    output_y = int(np.ceil(img_y / pool_size))

    windows = [img[y:y+pool_size, x:x+pool_size] for x in range(0, img_x, pool_size) for y in range(0, img_y, pool_size)]

    pooled = np.array([np.amax(window) for window in windows]).reshape(output_y, output_x)
    idxs = [np.argmax(window) for window in windows]

    mask = np.zeros((img_y, img_x))
    masked_windows = np.array([np.zeros(window.shape) for window in windows])

    for window, idx in zip(masked_windows, idxs):
        window.ravel()[idx] = 1

    w = 0
    for x in range(0, img_x, pool_size):
        for y in range(0, img_y, pool_size):
            mask[y:y+pool_size, x:x+pool_size] = masked_windows[w]
            w += 1
            
    return (pooled, mask)

funcs = {
    "relu":(ReLU, ReLU_prime),
    "sigmoid":(sigmoid, sigmoid_prime),
    "linear": (linear, linear_prime)
}


def main():
    a = np.arange(25).reshape(5, 5)
    print(a)
    pooled, mask = maxpool(a, 2)
    print("===")
    print(pooled)
    print("---")
    print(mask)



if __name__ == "__main__":
    main()