#modified from https://discuss.pytorch.org/t/a-gist-of-affine-transforms-in-pytorch/790/3

import numpy as np
from skimage.transform import warp, AffineTransform, SimilarityTransform, rotate, rescale
from imgaug import augmenters as iaa
import imgaug as ia


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        # img_data = np.array(img)
        # h, w, n_chan = img_data.shape
        h, w = img.shape
        scale_xy = np.random.uniform(*self.scale_range) # want to scale same in x and y
        # scale_x = np.random.uniform(*self.scale_range)
        # scale_y = np.random.uniform(*self.scale_range)
        # scale = (scale_x, scale_y)
        scale = (scale_xy, scale_xy)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        img = rotate(img, rotation, order=0)
        img = rescale(img, scale_xy, order=0)
        # af = AffineTransform(scale=scale, shear=shear, rotation=None, translation=translation)
        # st = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
        return img
        # return warp(img, af.inverse)
        # return  warp(img, st.inverse)
        # img1 = Image.fromarray(np.uint8(img_data1 * 255))
        # return img_data1


class CustomRandomTransform(object):
    def __init__(self):
        self.sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    def __call__(self, img1, img2):
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                self.sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by 0-10% of their height/width
                self.sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-180, 180), # rotate by -45 to +45 degrees
        #             shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0,1], # use nearest neighbour or bilinear interpolation (fast)
        #             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ))
            ],
            random_order=True)
        # images_aug = seq.augment_image(a)

        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        img1 = seq_det.augment_image(img1)
        img2 = seq_det.augment_image(img2)

        return img1, img2


class CustomRandomTransform_one_image(object):
    def __init__(self):
        self.sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    def __call__(self, img1):
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                self.sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by 0-10% of their height/width
                self.sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-180, 180), # rotate by -45 to +45 degrees
        #             shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0,1], # use nearest neighbour or bilinear interpolation (fast)
        #             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ))
            ],
            random_order=True)
        # images_aug = seq.augment_image(a)

        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        img1 = seq_det.augment_image(img1)

        return img1
