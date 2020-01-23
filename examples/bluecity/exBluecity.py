# Set working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# flowers
import logging
from PIL import Image as _Image

import pyglitch

import time

intervalGenerator0 = pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.LightnessPixelFunction(),
    lo=0,
    up=0.8
)

sortingFunction0 = pyglitch.StepPixelFunction(3)

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=0,
    reverse=False
)

logging.getLogger().setLevel(7)

I = sortFilter0.apply(_Image.open('bluecity.jpg'), mask=_Image.open('bluecity_mask.jpg')).save('bluecity_0.jpg')
