import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ...models import *

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()

    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = np.array(full_img)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    # HWC to CHW
    img_trans = img.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255
    
    img = torch.from_numpy(img_trans)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def predict(model_name, model_params, model_path, in_files, viz=True, scale_factor=1.0, out_threshold=0.5, save=True):
  model_list = {'CustomUNet': CustomUNet}
  unet = model_list[model_name](**model_params)#UNet(3, 1)
  print("Loading model {}".format(model_path))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device {device}')
  unet.to(device=device)
  unet.load_state_dict(torch.load(model_path, map_location=device))

  for i, fn in enumerate(in_files):
    print("\nPredicting image ...")

    img = Image.open(BytesIO(fn))

    mask = predict_img(net=unet, full_img=img, scale_factor=scale_factor, out_threshold=out_threshold, device=device)

    # if save:
    #     out_fn = out_files[i]
    #     result = mask_to_image(mask)
    #     result.save(out_files[i])

        # print("Mask saved to {}".format(out_files[i]))

    if viz:
        print("Visualizing results for image...")
        plot_img_and_mask(img, mask)