import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.data import coins as get_coins_image

def test_blur(gaussian_blur):

    img = get_coins_image().astype(np.float32) / 255
    blurred = gaussian_blur(img)
    expected = np.load('./presaved/gaussian_blur.npy')
    error = np.abs(blurred - expected)
    
    fig, ax = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    
    cmap = 'gray'
    ax[0].imshow(blurred, cmap=cmap)
    ax[1].imshow(expected, cmap=cmap)
    ax[2].imshow(error)
    
    ax[0].set_title('Yours')
    ax[1].set_title('Expected')
    ax[2].set_title('Max error: {:.2f}'.format(error.max()))
    
    for axis in ax:
        axis.set_axis_off()

    plt.show()


def test_grad(compute_grad):
    cmap = 'gray'

    img = get_coins_image().astype(np.float32) / 255
    dx, dy = compute_grad(img)

    fig, axes = plt.subplots(2, 3, constrained_layout=True, dpi=200)
    
    expected_dx = np.load('./presaved/grad_x.npy')
    expected_dy = np.load('./presaved/grad_y.npy')
    
    error_dx = np.abs(dx - expected_dx)
    error_dy = np.abs(dy - expected_dy)
    
    axes[0, 0].imshow(dx, cmap=cmap)
    axes[0, 1].imshow(expected_dx, cmap=cmap)
    axes[0, 2].imshow(error_dx)
    axes[0, 2].set_title('Max error: {:.2f}'.format(error_dx.max()))
    
    axes[0,0].set_title('Yours')
    axes[0,1].set_title('Expected')
    axes[1,0].set_title('Yours')
    axes[1,1].set_title('Expected')

    axes[1, 0].imshow(dy, cmap=cmap)
    axes[1, 1].imshow(expected_dy, cmap=cmap)
    axes[1, 2].imshow(error_dy)
    axes[1, 2].set_title('Max error: {:.2f}'.format(error_dy.max()))
    
    for ax in axes.flat:
        ax.set_axis_off()

    plt.show()


def test_direction(grad_direction):

    dx = np.load('./presaved/grad_x.npy')
    dy = np.load('./presaved/grad_y.npy')

    angles = grad_direction(dx, dy)
    expected = np.load('./presaved/angles.npy')
    error = np.abs(angles - expected)
    
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    
    cmap = 'gray'
    axes[0].imshow(angles, cmap=cmap)
    axes[1].imshow(expected, cmap=cmap)
    axes[2].imshow(error)

    axes[0].set_title('Yours')
    axes[1].set_title('Expected')
    axes[2].set_title('Max error: {:.2f}'.format(error.max()))
    
    for ax in axes:
        ax.set_axis_off()

    plt.show()



    
