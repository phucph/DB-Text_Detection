import math

import cv2
import imgaug
import numpy as np

from concern.config import State
from data.augmenter import AugmenterBuilder

from .data_process import DataProcess



class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get("augmenter_args")
        self.keep_ratio = kwargs.get("keep_ratio")
        self.resize_pad =  kwargs.get("resize_pad")
        self.only_resize = kwargs.get("only_resize")
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape["height"]
        width = resize_shape["width"]
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def resize_pad_image(self, img):
        height, width, _ = img.shape
        resize_shape = self.augmenter_args[0][1]
        _height = resize_shape["height"]
        _width = resize_shape["width"]
        width = min(width, max(int(height / img.shape[0] * img.shape[1] / 32 + 0.5) * 32, 32))
        canvas = np.zeros((_height,_width, 3), np.float32)
        if max(height, width) < resize_shape["height"]:
          image = cv2.resize(img, (width, height))
        else: 
          scale = resize_shape["width"]*1.0 / max(height, width)
          image = cv2.resize(img, dsize=None, fx=scale, fy=scale)
          height, width = image.shape[:2]
        canvas[:height, :width, :] = image
        return canvas

    def process(self, data):
        image = data["image"]
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.resize_pad:
                data["image"] = self.resize_pad_image(image)
            elif self.only_resize:
                data["image"] = self.resize_image(image)
            else:
                data["image"] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get("filename", data.get("data_id", ""))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data["is_training"] = True
        else:
            data["is_training"] = False
        return data


class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data["lines"]:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line["poly"]]
            else:
                new_poly = self.may_augment_poly(aug, shape, line["poly"])
            line_polys.append(
                {
                    "points": new_poly,
                    "ignore": line["text"] == "###",
                    "text": line["text"],
                }
            )
        data["polys"] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
