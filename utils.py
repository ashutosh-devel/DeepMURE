import matplotlib.pyplot as plt
import numpy as np


# def display_image_grid(list_of_images,n_rows,n_cols,new_fig=True,cmap='gray'):
#     if new_fig:
#         plt.figure()
#
#     for i in range(n_rows):
#         for j in range(n_cols):
#             idx = (i * n_cols) + j
#             plt.subplot(n_rows, n_cols, idx+1)
#             plt.imshow(list_of_images[idx],'gray')
#             plt.grid(False)
#             plt.axis('off')

def display_image_grid(list_of_images,n_rows,n_cols,new_fig=True,cmap='gray',mode='synth'):
    if new_fig:
        plt.figure()

    for i in range(n_rows):
        for j in range(n_cols):
            idx = (i * n_cols) + j
            plt.subplot(n_rows, n_cols, idx+1)
            plt.imshow(np.uint8((list_of_images[idx]*(255))),'gray')
            plt.grid(False)
            plt.axis('off')

def reverse_transform(img):
    img = (img * (256**2))**0.5 - 1
    return img

def real_display_image_grid(list_of_images,n_rows,n_cols,new_fig=True,cmap='gray',mode='synth'):
    if new_fig:
        plt.figure()

    for i in range(n_rows):
        for j in range(n_cols):
            idx = (i * n_cols) + j
            plt.subplot(n_rows, n_cols, idx+1)
            img = reverse_transform(list_of_images[idx]).numpy()
            img[np.where(img > 255.0)] = 255.0
            plt.imshow(np.uint8(img),'gray')
            plt.grid(False)
            plt.axis('off')

def mse_to_psnr(mse, peak_val=1.0):
    psnr = 10 * np.log10((peak_val**2) / mse)
    return psnr

def real_mse_to_psnr(mse, peak_val=1.0):
    psnr = 10 * np.log10((peak_val**2) / mse)
    return psnr

# def mse_to_psnr(mse, maxval=2.0):
#     return 10*np.log10((maxval**2)/(np.array(mse)))
