import matplotlib.pyplot as plt

import torchvision
import matplotlib.pyplot as plt

def show(tensors, fig_size=(10, 10)):
  try:
    tensors = tensors.detach().cpu()
  except:
    pass
  grid_tensor = torchvision.utils.make_grid(tensors)
  grid_image = grid_tensor.permute(1, 2, 0)
  plt.figure(figsize=fig_size)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(grid_image)

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()