import os
import glob
import numpy as np 
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image

processed_folder = 'processed'

trans = transforms.Compose([
	transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class PreProcessDataset(Dataset):
	"""docstring for PreProcessDataset"""
	def __init__(self, root, train=True, transform=trans):
		super().__init__()
		self.root = os.path.expanduser(root)
		self.train = train
		if self.train:
			self.train_source = os.path.join(self.root, 'train')
			self.train_dir = os.path.join(self.root, processed_folder, 'train')
			if not os.path.exists(self.train_dir):
				os.makedirs(self.train_dir)
				self._resize(self.train_source, self.train_dir)
			train_images = glob.glob((self.train_dir + '/*'))
			np.random.shuffle(train_images)
			self.train_image = list(train_images)
		else:
			self.test_source = os.path.join(self.root, 'test')
			self.test_dir = os.path.join(self.root, processed_folder, 'test')
			if not os.path.exists(self.test_dir):
				os.makedirs(self.test_dir)
				self._resize(self.test_source, self.test_dir)
			test_images = glob.glob((self.test_dir + '/*'))
			np.random.shuffle(test_images)
			self.test_image = list(test_images)
		
		self.transforms = transform

	@staticmethod
	def _resize(source_dir, target_dir):
		print(f'Start Resizing {source_dir} ')
		for i in tqdm(os.listdir(source_dir)):
			filename = os.path.basename(i)
			try:
				image = io.imread(os.path.join(source_dir, i))
				if len(image.shape) == 3 and image.shape[-1] == 3:
					H, W, _ = image.shape
					if H < W:
						ratio = W / H
						H = 512
						W = int(ratio*H)
					else:
						ratio = H / W
						W = 512
						H = int(ratio*W)
					image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
					io.imsave(os.path.join(target_dir, filename), image)
			except:
				continue

	def __len__(self):
		if self.train:
			return len(self.train_dir)
		else:
			return len(self.test_dir)

	def __getitem__(self, index):
		if self.train:
			image, target = self.train_image[index], 1
			image = Image.open(image)
			#style_image = Image.open(style_image)
		else:
			image, target = self.test_image[index], 1
			image = Image.open(image)
		
		if self.transforms:
			image = self.transforms(image)
			#style_image = self.transforms(style_image)
			return image

