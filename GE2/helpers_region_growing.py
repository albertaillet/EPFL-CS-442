import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=2)
    
    resize_width = 400 
    resize_height = int(image.shape[0] * resize_width / image.shape[1])
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    
    image = image.astype(np.float32) / 255
    
    return image

def display_init(image, seeds):
    j, i = seeds.T
    
    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=150)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image {image.shape}')
    
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # plot the seeds, each with a separate color
    colors = ['red', 'yellow', 'blue']
    for n in range(seeds.shape[0]):
        axes[1].scatter([j[n]], [i[n]], color=colors[n])

    axes[1].imshow(image)
    axes[1].set_title('Original Image with Seeds')

def display(image, title):
    fig, axes = plt.subplots(constrained_layout=True, dpi=150)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.imshow(image)
    axes.set_title(title)

def test_seg(image, image_orig, seeds, gt_path):
    gt = np.loadtxt(gt_path)
    
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=150)

    for axis in axes:
        axis.set_axis_off()

    axes[0].imshow(image_orig)
    axes[0].set_title('Original')

    axes[1].imshow(image)
    mIoU = compute_iou(image, gt)
    axes[1].set_title(f'Yours, mIoU {mIoU}%')
    
    axes[2].imshow(gt)
    axes[2].set_title('Expected')

    j, i = seeds.T
    # plot the seeds, each with a separate color
    colors = ['red', 'yellow', 'blue']
    for n in range(seeds.shape[0]):
        for ax in axes:
            ax.scatter([j[n]], [i[n]], color=colors[n])


def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    labels = np.unique(y_true).reshape(-1, 1)
    
    intersection = ((y_true == labels) & (y_pred == y_true)).sum(axis=-1)
    ground_truth_set = (labels == y_true).sum(axis=-1)
    predicted_set = (labels == y_pred).sum(axis=-1)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return 100 * np.mean(IoU)