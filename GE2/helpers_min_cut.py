import numpy as np
import matplotlib.pyplot as plt

def add_source_and_sink(graph, list_source, list_sink,thresh=float('Inf')):
    graph['source']={}
    graph['sink']={}
    for idx, v in enumerate(list_source):
        if idx==0:
            graph['source']={}
            
        graph['source'].update({v: thresh})
        graph[v].update({'source': thresh})
        
    for idx, v in enumerate(list_sink):
        if idx==0:
            graph['sink']={}
            
        graph['sink'].update({v: thresh})
        graph[v].update({'sink': thresh})
        
        
        
def graph_to_binary_image(graph, image_shape):
    bin_img = np.ones(image_shape)
    del graph['source']
    del graph['sink']
    
    for pix in graph:
        if graph[pix]:
            bin_img[pix] = 0
    return bin_img


def plotting_segmentation(img, bin_img, s, t):
    img_rgb = np.array([img,img,img])
    img_rgb = np.transpose(img_rgb, (1,2,0))

    proportion = 0.7
    img_rgb[:, :, 1] = proportion*img_rgb[:, :,1]+(1-proportion)*bin_img
    img_rgb[:, :, 2] = proportion*img_rgb[:, :,2]+(1-proportion)*bin_img

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(1,3, 1)
    ax1.set_axis_off()
    ax1.set_title("Original image")
    ax1.imshow(img, cmap='gray')
    
    ax2 = fig.add_subplot(1,3, 2)
    ax2.set_axis_off()
    ax2.set_title("Min-Cut as a binary mask")
    ax2.imshow(bin_img, cmap='gray')
    
    ax3 = fig.add_subplot(1,3, 3)
    ax3.set_axis_off()
    ax3.set_title("Superposition of original and mask")
    ax3.imshow(img_rgb, cmap='gray')
    
    
    ax1.scatter([i[1] for i in s], [i[0] for i in s], c='green')
    ax1.scatter([i[1] for i in t], [i[0] for i in t], c='red')
    ax2.scatter([i[1] for i in s], [i[0] for i in s], c='green')
    ax2.scatter([i[1] for i in t], [i[0] for i in t], c='red')
    ax3.scatter([i[1] for i in s], [i[0] for i in s], c='green')
    ax3.scatter([i[1] for i in t], [i[0] for i in t], c='red')