import os
from pathlib import Path

import cv2
import numpy as np

from .image_utils import load_images


class LiteClassifier:
    def __init__(self, model_path: Path):
        model_path = model_path.resolve().absolute()
        if not model_path.exists():
            raise Exception('Cannot find model file')

        self.lite_model = cv2.dnn.readNet(model_path)

    def classify(self, image_paths, size=(256, 256)):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        result = {}
        for image_path in image_paths:
            loaded_images, _ = load_images([image_path], size, image_names=[image_path])
            loaded_images = np.rollaxis(loaded_images, 3, 1)

            self.lite_model.setInput(loaded_images)
            pred = self.lite_model.forward()

            result[image_path] = {
                "unsafe": pred[0][0],
                "safe": pred[0][1],
            }

        return result
