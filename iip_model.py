import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid


from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision.transforms.functional import to_pil_image

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE=128
MASK_SIZE=32

random.seed(42)
import warnings
warnings.filterwarnings("ignore")

latent_dim = 100
# size of each image dimension
img_size = 128
# size of random mask
mask_size = 32
# number of image channels
channels = 3
#Generator model 
GEN_MODEL_PATH='generator.pth'


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=mask_size, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]


    def apply_random_mask(self, img):
        """Randomly masks image"""

        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
    
        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train" or self.mode=="val":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
    
class GrayscaleTo3Channels(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # If the image is grayscale ('L'), convert to 3 channels (RGB)
        if img.mode == 'L':
            img = img.convert('RGB')
        return img



transforms_ = transforms.Compose( [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    GrayscaleTo3Channels(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.encoder = nn.Sequential(
            *downsample(channels, 64, normalize=False),  # 128x128 → 64x64
            *downsample(64, 128),  # 64x64 → 32x32
            *downsample(128, 256),  # 32x32 → 16x16
            *downsample(256, 512),  # 16x16 → 8x8
            *downsample(512, 512),  # 8x8 → 4x4
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            *upsample(512, 512),  # 4x4 → 8x8
            *upsample(512, 256),  # 8x8 → 16x16
            *upsample(256, 128),  # 16x16 → 32x32
            nn.Conv2d(128, channels, 3, 1, 1),  # 32x32 output
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize=True):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]  # Kernel 4x4
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
        *discriminator_block(channels, 64, stride=2, normalize=False),  # 128x128 -> 64x64
        *discriminator_block(64, 128, stride=2),  # 64x64 -> 32x32
        *discriminator_block(128, 256, stride=2),  # 32x32 -> 16x16
        *discriminator_block(256, 512, stride=2),  # 16x16 -> 8x8
        *discriminator_block(512, 512, stride=1, normalize=False),  # ❌ Disable normalization here
        nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)  # Final 8x8 validity map
    )


    def forward(self, img):
        return self.model(img)  # Output shape: (batch_size, 1, 8, 8)




generator=Generator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(GEN_MODEL_PATH, map_location=torch.device('cpu'))
generator.load_state_dict(state_dict)
generator.to(device)  # Don't forget to move model to device!
generator.eval() 


# File uploader
uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if 'coordinates' not in st.session_state:
    st.session_state['coordinates'] = None



if uploaded_file:
    # Open the uploaded image file
    img = Image.open(uploaded_file)
    print(img.size)

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    draw = ImageDraw.Draw(img)


    if st.session_state["coordinates"]:
        coords = st.session_state["coordinates"]
        draw.rectangle(coords, fill=None, outline="red")

    value = streamlit_image_coordinates(img, key="rectangle", click_and_drag=True)
    st.write(value)

    #print('value:', value)

    # Assume this is the drag selection with x1/y1
    if value is not None:
        x1, y1 = value["x1"], value["y1"]
        point1 = (x1, y1)
        point2 = (x1 + MASK_SIZE, y1 + MASK_SIZE)

        bounding_box = (x1, y1, x1 + MASK_SIZE, y1 + MASK_SIZE)
        print("Bounding box:", bounding_box)

        mask = img.crop(bounding_box)
        st.image(mask)

        draw1 = ImageDraw.Draw(img)
        draw1.rectangle([point1, point2], fill='white')

        # DO NOT overwrite `value` — use a new variable if you must call again
        click_coords = streamlit_image_coordinates(img, key="rect")
        auximg = img

        masked_img = auximg
        masked_imgs = transforms_(masked_img).unsqueeze(0).to(device)

        with torch.no_grad():
            gen_part = generator(masked_imgs)

        # Denormalize generator output
        gen_part = (gen_part + 1) / 2.0
        pred_part = to_pil_image(gen_part.squeeze(0).cpu())

        # Paste into a copy of the masked image
        final_img = masked_img.copy()
        final_img.paste(pred_part, (x1, y1))

        st.image(pred_part, caption="Predicted Patch")
        st.image(final_img, caption="Final Inpainted Image")

        if st.session_state["coordinates"] != (point1, point2):
            st.session_state["coordinates"] = (point1, point2)
            st.rerun()

    # Still show the selected coordinates
    if st.session_state["coordinates"]:
        coords = st.session_state["coordinates"]


## use auximg, mask to get predicted region
## predict mask image
## overlay predicted region at boundingbox coordinaetes with predicted image
## img, aux(masked image), mask

## pred = generator(aux)

