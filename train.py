from imutils import paths
import torch
from torch.utils.data import Dataset, DataLoader
import time
import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import numpy as np
import math
import random
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision import transforms
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt
from os.path import splitext
from os import listdir
from glob import glob
import logging
from PIL import ImageOps
import argparse
import sys
from torch import optim
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tempfile import TemporaryFile
outfile = TemporaryFile()

!echo "$(pip freeze | grep albumentations) is successfully installed"

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = '/JPG Images/'
RAW_IMAGE_DATASET_PATH = '/Raw Images/'
MASK_DATASET_PATH = '/VOC_Outputs/SegmentationClassPNG'
# define the test split
TEST_SPLIT = 0.25
# determine the device to be used for training and evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CLASSES = 10
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 4
# BATCH_SIZE = 64
BATCH_SIZE = 4
# BATCH_SIZE = 5

# define the input image dimensions
INPUT_IMAGE_WIDTH = 1908
INPUT_IMAGE_HEIGHT = 969

# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"

# # define the path to the output serialized model, model training
# # plot, and testing image paths
MODEL_PATH = "/unet_tgs_salt.pth"
PLOT_PATH = "/plot.png"
TEST_PATHS = "/test_paths.txt"

LABEL_TO_COLOR = {0:[0,0,0], 1:[128,0,0], 2:[0,128,0], 3:[128,128,0], 4:[0,0,128], 5:[128,0,128], 6:[0,128,128], 7:[128,128,128], 8:[64,0,0], 9:[192,0,0]}
def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))
    for k,v in LABEL_TO_COLOR.items():
        mask[np.all(rgb==v, axis=2)] = k
    return mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mask = sample["image"], sample["mask"]  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels
        mask = rgb2mask(np.array(mask))
        sample = {'image': torch.from_numpy(image).float(),
                  'mask': torch.from_numpy(mask).long(), 
                 }
        return sample

class SegmentationTrainingDataset(Dataset):
	def __init__(self, images, masks, transforms):
		# store the image, and mask filepaths, and augmentation
		# transforms
		self.images = images
		self.masks = masks
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.images)
	def __getitem__(self, idx):
		# grab the image path from the current index
		image = self.images[idx]
		mask = self.masks[idx]
		# check to see if we are applying any transformations
		image = transforms.Compose([transforms.ToPILImage()])(image)
		mask = transforms.Compose([transforms.ToPILImage()])(mask)	 
		sample = {"image": image, "mask": mask}
		if self.transforms:
			sample = self.transforms(sample)
		return (sample["image"], sample["mask"])

# TOD0: png to jpg
class SegmentationTestingDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
		image = cv2.createCLAHE(clipLimit = 5).apply(image) + 30
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		mask = cv2.imread(self.maskPaths[idx])
	
		image = transforms.Compose([transforms.ToPILImage()])(image)
		mask = transforms.Compose([transforms.ToPILImage()])(mask)
		# check to see if we are applying any transformations
		sample = {"image":image, "mask":mask}
		if self.transforms:
			sample = self.transforms(sample)
		return (sample["image"], sample["mask"])

class AugmentedDataset(Dataset):
    def __init__(self, images_filepaths, masks_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.masks_filepaths = masks_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.createCLAHE(clipLimit = 5).apply(image) + 30
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.masks_filepaths[idx])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            mask = self.transform(image=mask)["image"]
        return (image, mask)

class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures

class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=10, retainDim=True,
		 outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
	def forward(self, x):
		# grab the features from the encoder
		encFeatures = self.encoder(x)
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		# return the segmentation map
		return map

imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
rawImagePaths = sorted(list(paths.list_images(RAW_IMAGE_DATASET_PATH)))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing

split = train_test_split(imagePaths, maskPaths,	test_size=TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

def visualize_augmentations(dataset, idx=0, samples=20, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, mask = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show() 

aug_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=5, p=0.5),
        A.MedianBlur(blur_limit=7, always_apply=False, p=0.5)
    ]
)
train_dataset = AugmentedDataset(images_filepaths=trainImages, masks_filepaths=trainMasks, transform=aug_transform)

random.seed(42)
visualize_augmentations(train_dataset)

def generate_augmented_image(dataset, idx, samples=6):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    img = []
    mk = []
    for idx in range(samples):
      image, mask = dataset[idx] 
      img.append(image)
      mk.append(mask)
    return img, mk


random.seed(42)
finalized_train_images = []
finalized_train_masks = []
for idx,image in enumerate(trainImages):
  img, mk = generate_augmented_image(train_dataset, idx)
  finalized_train_images += img
  finalized_train_masks += mk


# define transformations
tf = transforms.Compose([ToTensor()])
# create the train and test datasets
trainDS = SegmentationTrainingDataset(images=finalized_train_images, masks=finalized_train_masks, transforms=tf)
testDS = SegmentationTestingDataset(imagePaths=testImages, maskPaths=testMasks, transforms=tf)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size= BATCH_SIZE, pin_memory= PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size= BATCH_SIZE, pin_memory= PIN_MEMORY,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(DEVICE)
# initialize loss function and optimizer
lossFunc = nn.CrossEntropyLoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // BATCH_SIZE
testSteps = len(testDS) // BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		print("index ",i)
		# send the input to the device
		(x, y) = (x.to(device=DEVICE, dtype=torch.float32), y.to(device=DEVICE, dtype=torch.long))
		# perform a forward pass and calculate the training loss
		x = x.permute(0, 3, 1, 2)
		pred = unet(x)	
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(device=DEVICE, dtype=torch.float32), y.to(device=DEVICE, dtype=torch.long))
			# make the predictions and calculate the validation loss
			x = x.permute(0, 3, 1, 2)
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	print(totalTestLoss, testSteps)
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)
# serialize the model to disk
torch.save(unet, MODEL_PATH)


# USAGE
# python predict.py
# import the necessary packages
def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

from torchvision import transforms as tfs
def make_predictions(model, imagePath, maskPath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		# resize the image and make a copy of it for visualization
		# image = cv2.resize(image, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		# filename = imagePath.split(os.path.sep)[-1]
		# groundTruthPath = os.path.join(MASK_DATASET_PATH,	filename)
		# print(groundTruthPath)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(maskPath)
		gtMask = cv2.cvtColor(gtMask, cv2.COLOR_BGR2RGB)
		# gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))
    # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(device=DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = F.softmax(predMask, dim=1)[0]
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(MODEL_PATH).to(DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
  id = path.split('_')[0].split('/')[-1]
  maskPath = [s for s in maskPaths if id in s][0]
  # make predictions and visualize the results
  make_predictions(unet, path, maskPath)


