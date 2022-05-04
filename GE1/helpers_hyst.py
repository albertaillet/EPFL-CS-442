import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_label_pixels(label_pixels):
    img = cv2.imread('./presaved/coins_gradient.png', 0)[50:100, 50:100]
    pred = label_pixels(img)

    expected = np.load('./presaved/label_pixels.npy')[50:100, 50:100]

    error = np.abs( pred - expected )
    
    fig, axes = plt.subplots(1, 4, constrained_layout=True, dpi=200)
    
    cmap = 'magma'
    axes[0].imshow(img, cmap=cmap)
    axes[1].imshow(pred, cmap=cmap)
    axes[2].imshow(expected, cmap=cmap)
    axes[3].imshow(error)

    axes[0].set_title('Original')
    axes[1].set_title('Yours')
    axes[2].set_title('Expected')
    axes[3].set_title('Max error: {:.2f}'.format(error.max()))
    
    for ax in axes:
        ax.set_axis_off()

    plt.show()


def test_update(update):
    labeled = np.load('./presaved/label_pixels.npy')[80:90, 55:65]
    
    point1 = [3,0]
    point2 = [9,4]
    point3 = [9,7]
    pred1 = update(labeled.copy(), point1[0], point1[1])
    pred2 = update(labeled.copy(), point2[0], point2[1])
    pred3 = update(labeled.copy(), point3[0], point3[1])
    
    fig, axes = plt.subplots(1, 4, figsize=(18,5))
    
    cmap = 'magma'
    axes[0].imshow(labeled, cmap=cmap)
    axes[1].imshow(pred1, cmap=cmap)
    axes[2].imshow(pred2, cmap=cmap)
    axes[3].imshow(pred3, cmap=cmap)

    axes[0].set_title('Original')
    axes[1].set_title(f'{point1} (2 -> 2)')
    axes[2].set_title(f'{point2} (0 -> 0)')
    axes[3].set_title(f'{point3} (1 -> 2)')

    axes[1].scatter(*point1[::-1])
    axes[2].scatter(*point2[::-1])
    axes[3].scatter(*point3[::-1])

    plt.show()
