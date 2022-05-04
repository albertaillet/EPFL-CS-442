import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_gauss_1d(gausian_kernel_1d_func):
    
    ### test settings
    size = 11
    sigmas = [0.1, 0.5, 1, 3, 5, 7, 9, 11]
    
    ### presaved kernels
    k_saved = np.load('./presaved/gauss_1d.npy', allow_pickle=True).item()

    fig, ax = plt.subplots(1,3, figsize=(15,5)) 
#     plt.subplots_adjust(hspace=0.1,wspace=0.5)
    
    for sigma in sigmas:
        k_pred = gausian_kernel_1d_func(size, sigma)
        ax[0].plot(k_pred, label=str(sigma))
        
        k_saved_ = k_saved[sigma]
        ax[1].plot(k_saved_, label=str(sigma))
    
        ax[2].plot(np.abs(k_saved_-k_pred), label=str(sigma))
        
    
    ax[0].set_title('Yours')
    ax[1].set_title('Expected')
    ax[2].set_title('Error (flat line means 0 error)')
    
    for axis in ax:
        axis.legend()

    plt.show()


def test_gauss_2d(gausian_kernel_2d_func):

    cmap = 'viridis'
    d = np.load('./presaved/gauss_2d.npy', allow_pickle=True).item()
    
    sizes = [3,5,31]
    sigmas = [1,3]

    fig, ax = plt.subplots(6, 3, figsize=(10,20)) 
    i = 0
    for size in sizes:
        for sigma in sigmas:
            gaus_1d, gaus_2d = d[size][sigma]
            gaus_2d_pred = gausian_kernel_2d_func(gaus_1d)

            error = np.abs(gaus_2d_pred - gaus_2d)

            ax[i, 0].imshow(gaus_2d_pred, cmap=cmap)
            ax[i, 1].imshow(gaus_2d, cmap=cmap)
            ax[i, 2].imshow(error, cmap=cmap)

            for k in range(3):
                ax[i,k].set_xticks([])
                ax[i,k].set_yticks([])

            ax[i,0].set_ylabel(f'size {size}, sigma {sigma}')
            ax[i,1].set_ylabel(f'size {size}, sigma {sigma}')

            i += 1

    ax[0,0].set_title('Yours')
    ax[0,1].set_title('Expected')
    ax[0,2].set_title('Error (difference between the two)')

    plt.show()


def test_strided_coins(conv_with_stride):

    img = cv2.imread('./presaved/coins.png', 0)
    ## Example of applying a 2d filter to an image
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    R = conv_with_stride(img, kernel)
    expected_result = np.load('./presaved/coins_strided.npy')

    try:
        error = np.abs(expected_result - R) 
    except ValueError: # in cases sizes are inconsistent
        error = expected_result.copy()

    ## Visulization 
    fig, ax = plt.subplots(1, 3, figsize=(15,5)) 

    ax[0].imshow(R)
    ax[1].imshow(expected_result)
    ax[2].imshow(error)

    ax[0].set_title('Yours')
    ax[1].set_title('Expected')
    ax[2].set_title('Max error: {:.2f}'.format(error.max()))

    plt.show()


def test_thresholding(thresholding):
    grad_mag = np.load('./presaved/grad_magnitude.npy')
    
    gt = np.load('./presaved/thresholded_200.npy', allow_pickle=True)
    threshold = 200

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    pred = thresholding(grad_mag, threshold)
    error = np.abs(pred - gt)

    ax[0].imshow(gt, cmap='jet')
    ax[1].imshow(pred, cmap='jet')
    ax[2].imshow(error, cmap='jet')

    ax[0].set_title('Yours')
    ax[1].set_title('Expected')
    ax[2].set_title('Max error: {:.2f}'.format(error.max()))

    for i in range(3):
        ax[i].set_xticks([])

    for i in range(2):
        ax[0].set_ylabel(f'thresh {threshold}')
        ax[1].set_ylabel(f'thresh {threshold}')

    plt.show()


def test_canny():

    img = cv2.imread('./presaved/coins.png', 0)
    grad_mag = np.load('./presaved/grad_magnitude.npy')
    gt = np.load('./presaved/thresholded_200.npy', allow_pickle=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(15,5)) 
    
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(grad_mag, cmap='jet')
    ax[2].imshow(gt, cmap='jet')

    ax[0].set_title('Original')
    ax[1].set_title('Gradient Magnitude')
    ax[2].set_title('Gradient Thresholded')

    for i in range(3):
        ax[i].set_axis_off()

    plt.show()