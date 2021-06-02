from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "leaf": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: leaf_loader(folder),
    }
}


def leaf_loader(folder):
    img = open_file(folder + "leaf.mat")
    img = img["leaf"]

    rgb_bands = (43, 21, 11)  # AVIRIS sensor

    gt = open_file(folder + "leaf_gt.mat")["leaf_gt"]
    label_values = [
            "Undefined",
            "lowest",
            "lower",
            "middle",
            "high",
        ]

    ignored_labels = [0]
#     ignored_labels = []
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
