import numpy as np
import matplotlib.pyplot as plt

def display(I):
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    axes[0].imshow(I[:,:,0],cmap='gray')
    axes[0].set_title('Light 1')
    axes[1].imshow(I[:,:,1],cmap='gray')
    axes[1].set_title('Light 2')
    axes[2].imshow(I[:,:,2],cmap='gray')
    axes[2].set_title('Light 3')
    
    for ax in axes:
        ax.set_axis_off()
        
def show_blur(I_blur):
    expected_I_blur = np.load('presaved/I_blur.npy') 
    
    error = np.abs(I_blur- expected_I_blur)
    
    fig, axes = plt.subplots(3, 3, constrained_layout=True, dpi=200)
    
    axes[0, 0].imshow(I_blur[:,:,0], cmap='gray')
    axes[0, 0].set_title('Your 1')
    axes[0, 1].imshow(expected_I_blur[:,:,0], cmap='gray')
    axes[0, 1].set_title('Expected 1')
    axes[0, 2].imshow(error[:,:,0], cmap='gray')
    axes[0, 2].set_title('Max error: {:.2f}'.format(error[:,:,0].max()))
    
    axes[1, 0].imshow(I_blur[:,:,1], cmap='gray')
    axes[1, 0].set_title('Your 2')
    axes[1, 1].imshow(expected_I_blur[:,:,1], cmap='gray')
    axes[1, 1].set_title('Expected 2')
    axes[1, 2].imshow(error[:,:,1], cmap='gray')
    axes[1, 2].set_title('Max error: {:.2f}'.format(error[:,:,1].max()))
    
    axes[2, 0].imshow(I_blur[:,:,2], cmap='gray')
    axes[2, 0].set_title('Your 3')
    axes[2, 1].imshow(expected_I_blur[:,:,2], cmap='gray')
    axes[2, 1].set_title('Expected 3')
    axes[2, 2].imshow(error[:,:,2], cmap='gray')
    axes[2, 2].set_title('Max error: {:.2f}'.format(error[:,:,2].max()))
    
    for ax in axes.flat:
        ax.set_axis_off()
        
def check_flatten(I_flatten):
    expected_I_flatten = np.load('presaved/I_flatten.npy')
    print('The expected size is: ', expected_I_flatten.shape, ' Yours is: ', I_flatten.shape)
    try:
        error = np.abs(expected_I_flatten - I_flatten)
        print('Max error: {:.2f}'.format(error.max()))
    except:
        print('Size mismatching!')
        
def show_M(M):
    M = M.T.reshape(256, 256, 3)
    M_show = M.mean(axis=-1)
    M_show = M_show-M_show.min()/(M_show.max()-M_show.min())*255
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    
    expected_M = np.load('presaved/M.npy')
    expected_M = expected_M.T.reshape(256, 256, 3)
    expected_M_show = expected_M.mean(axis=-1)
    expected_M_show = expected_M_show-expected_M_show.min()/(expected_M_show.max()-expected_M_show.min())*255
    
    error_M = np.abs(M - expected_M)
    
    axes[0].imshow(M_show, cmap='gray')
    axes[0].set_title('Your M')
    axes[1].imshow(expected_M_show, cmap='gray')
    axes[1].set_title('Expected M')
    axes[2].imshow(error_M, cmap='gray')
    axes[2].set_title('Max error: {:.2f}'.format(error_M.max()))
    
    
    for ax in axes.flat:
        ax.set_axis_off()
        
def show_albedo_normal(albedo, normal):
    
    albedo = albedo.T.reshape(256, 256)
    normal = normal.T.reshape(256, 256, 3)
    normal_show = normal + 1
    normal_show = normal_show/2*255
    normal_show = normal_show.astype(np.uint8)
    
    expected_albedo = np.load('presaved/albedo.npy')
    expected_albedo = expected_albedo.T.reshape(256, 256)
    expected_normal = np.load('presaved/normal.npy')
    expected_normal = expected_normal.T.reshape(256, 256, 3)
    expected_normal_show = expected_normal + 1
    expected_normal_show = expected_normal_show/2*255
    expected_normal_show = expected_normal_show.astype(np.uint8)
   
    error_albedo = np.abs(albedo - expected_albedo)
    error_normal = np.abs(normal - expected_normal)
    
    
    fig, axes = plt.subplots(2, 3, constrained_layout=True, dpi=200)
    
    axes[0, 0].imshow(albedo, cmap='gray')
    axes[0, 0].set_title('Your albedo')
    axes[0, 1].imshow(expected_albedo, cmap='gray')
    axes[0, 1].set_title('Expected albedo')
    axes[0, 2].imshow(error_albedo, cmap='gray')
    axes[0, 2].set_title('Albedo max error: {:.2f}'.format(error_albedo.max()))
    axes[1, 0].imshow(normal_show)
    axes[1, 0].set_title('Your normal')
    axes[1, 1].imshow(expected_normal_show)
    axes[1, 1].set_title('Expected normal')
    axes[1, 2].imshow(error_normal, cmap='gray')
    axes[1, 2].set_title('Normal max error: {:.2f}'.format(error_normal.max()))
    
    for ax in axes.flat:
        ax.set_axis_off()
            
def show_mask(mask):
    mask = mask.reshape(256, 256)
    expected_mask = np.load('presaved/mask.npy').reshape(256, 256)
    
    error_mask = np.abs(mask - expected_mask)
    
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Your mask')
    axes[1].imshow(expected_mask, cmap='gray')
    axes[1].set_title('Expected mask')
    axes[2].imshow(error_mask, cmap='gray')
    axes[2].set_title('Max error: {:.2f}'.format(error_mask.max()))
    
    
    for ax in axes.flat:
        ax.set_axis_off()
        
def get_mask(array, threshold=30):
    mask = array.copy()
    mask[array<=threshold] = 0
    mask[array>threshold] = 1
    return mask

def show_final_results(albedo, normal, threshold=30):
    mask = get_mask(albedo, threshold=threshold)
    normal_vis_clean = np.zeros(normal.shape)
    for i in range(len(mask)):
        if not mask[i]:
            continue

        normal_vis_clean[:, i] = normal[:, i]+1
            
    albedo = albedo.reshape(256, 256)
    normal_vis = normal.T.reshape(256, 256, 3)+1
    normal_vis = normal_vis/2*255
    normal_vis = normal_vis.astype(np.uint8)
    normal_vis_clean = normal_vis_clean.T.reshape(256, 256, 3)
    normal_vis_clean = normal_vis_clean/2*255
    normal_vis_clean = normal_vis_clean.astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, constrained_layout=True, dpi=200)
    axes[0].imshow(albedo,cmap='gray')
    axes[0].set_title('Albedo')
    axes[1].imshow(normal_vis)
    axes[1].set_title('Normal')
    axes[2].imshow(normal_vis_clean)
    axes[2].set_title('Normal - Cleaned')
    
    for ax in axes:
        ax.set_axis_off()
