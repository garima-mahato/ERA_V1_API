import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .utility import *

class GradCam(object):

	def __init__(self, model, target_layers, num_classes):
		super(GradCam, self).__init__()
		self.model = model
		self.target_layers = target_layers
		self.num_classes = num_classes
		self.device = next(model.parameters()).device

		self.activations_map = {}
		self.gradients_map = {}

		self.model.eval()
		self.register_hooks()

	def register_hooks(self):
		def _wrap_forward_hook(layer_name):
			def _forward_hook(module, input, output):
				self.activations_map[layer_name] = output.detach()
			return _forward_hook

		def _wrap_backward_hook(layer_name):
			def _backward_hook(module, grad_out, grad_in):
				self.gradients_map[layer_name] = grad_out[0].detach()
			return _backward_hook

		for name, module in self.model.named_modules():
			if name in self.target_layers:
				module.register_forward_hook(_wrap_forward_hook(name))
				module.register_backward_hook(_wrap_backward_hook(name))

	def make_one_hots(self, target_class=None):
		one_hots = torch.zeros_like(self.output)
		if target_class:
			ids = torch.LongTensor([[target_class]] * self.batch_size).to(self.device)
			one_hots.scatter_(1,ids,1.0)
		else:
			one_hots = torch.zeros((self.batch_size, self.num_classes)).to(self.device)
			for i in range(len(self.pred)):
			  one_hots[i][self.pred[i][0]] = 1.0
		return one_hots

	def forward(self, data):
		self.batch_size, self.img_ch, self.img_h, self.img_w = data.shape
		data = data.to(self.device)
		self.output = self.model(data)
		self.pred = self.output.argmax(dim=1, keepdim=True)

	def backward(self, target_class=None):
		one_hots = self.make_one_hots(target_class)
		self.model.zero_grad()
		self.output.backward(gradient=one_hots, retain_graph=True)

	def __call__(self, data, target_layers, target_class=None):
		self.forward(data)
		self.backward(target_class)

		output = self.output
		saliency_maps = {}
		for target_layer in target_layers:
			activations = self.activations_map[target_layer]	#[64, 512, 4, 4]
			grads = self.gradients_map[target_layer]	#[64, 512, 4, 4]
			weights = F.adaptive_avg_pool2d(grads, 1)	#[64, 512, 1, 1]

			saliency_map = torch.mul(activations, weights).sum(dim=1, keepdim=True)	
			saliency_map = F.relu(saliency_map)	#[64,1,4,4]
			saliency_map = F.interpolate(saliency_map, (self.img_h, self.img_w), mode="bilinear", align_corners=False)	#[64,1,32,32]

			saliency_map = saliency_map.view(self.batch_size, -1)
			saliency_map -= saliency_map.min(dim=1, keepdim=True)[0]
			saliency_map /= saliency_map.max(dim=1, keepdim=True)[0]
			saliency_map = saliency_map.view(self.batch_size, 1,self.img_h, self.img_w)
			saliency_maps[target_layer] = saliency_map

		return saliency_maps, self.pred

class VisualizeCam(object):

	def __init__(self, model, classes, target_layers):
		super(VisualizeCam, self).__init__()
		self.model = model
		self.classes = classes
		self.target_layers = target_layers
		self.device = next(model.parameters()).device

		self.gcam = GradCam(model, target_layers, len(classes))
		
	def visualize_cam(self, mask, img):
	    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
	    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
	    b, g, r = heatmap.split(1)
	    heatmap = torch.cat([r, g, b])
	    
	    result = heatmap+img.cpu()
	    result = result.div(result.max()).squeeze()
	    return heatmap, result

	def plot_heatmaps(self, img_data, target_class, img_name, nrows=2, ncols=5, figsize_height=10, figsize_width=4):
		fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_height, figsize_width), subplot_kw={'xticks': [], 'yticks': []})
		fig.suptitle('GradCam for class: %s' %target_class, fontsize=13, weight='medium', y=1.05)

		for ax, data in zip(axs.flat, img_data):
			img = data["img"]
			npimg = img.cpu().numpy()
			ax.imshow(np.transpose(npimg, (1, 2, 0)))
			ax.set_title("%s" % (data["label"]))

		plt.savefig(img_name)

	def plot_img_heatmap(self, images, target_layers, PATH, target_inds=None, metric="", name="fig", columns=5, figsize_height=10, figsize_width=15):
		pred_images = []
		for i in range(len(images)):
			pred_images.append(torch.as_tensor(images[i]["img"]))
		pred_images = torch.stack(pred_images)
		masks_map, pred = self.gcam(pred_images, target_layers, target_inds)
		
		rows = math.ceil(len(images) / columns)
		fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(figsize_width, figsize_height), subplot_kw={'xticks': [], 'yticks': []})
		fig.suptitle('GradCam for %s misclassified images' %len(images), fontsize=15, weight='medium', y=1.05)
		fig.subplots_adjust(hspace = 0.5)
		
		i = 0
		for ax,_ in zip(axs.flat, images):
			img = images[i]["img"]
			mask = masks_map[target_layers[len(target_layers)-1]][i]
			heatmap, _ = self.visualize_cam(mask, img)
			img = denormalize(img)
			img = np.transpose(img.cpu().numpy(), (1, 2, 0))
			heatmap = np.transpose(heatmap.cpu().numpy(), (1, 2, 0))
			superimposed_img = cv2.addWeighted(img, 1.0, heatmap, 0.4, 0)
			ax.imshow(superimposed_img)
			ax.set_title(f"{i+1}) Ground Truth: {self.classes[images[i]['target']]},\n Prediction: {self.classes[images[i]['pred']]}", fontsize=8)
			i = i + 1

		plt.savefig(PATH+"/"+str(name)+".png")	

	def __call__(self, images, target_layers, PATH, target_inds=None, metric=""):
		masks_map, pred = self.gcam(images, target_layers, target_inds)
		for i in range(min(len(images),5)):
			img = images[i]
			results_data = [{
				"img": denormalize(img),
				"label": "Result:"
			}]
			heatmaps_data = [{
				"img": denormalize(img),
				"label": "Heatmap:"
			}]
			for layer in target_layers:
				mask = masks_map[layer][i]
				heatmap, result = self.visualize_cam(mask, img)
				results_data.append({
					"img": denormalize(result),
					"label": layer
				})
				heatmaps_data.append({
					"img": heatmap,
					"label": layer
				})
			pred_class = self.classes[pred[i][0]]
			fname = PATH + "/gradcam_%s_%s_%s.png" % (metric, i, pred_class)
			self.plot_heatmaps(results_data+heatmaps_data, pred_class, fname)