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
    pyglitch.HuePixelFunction(),
    lo=154/360,
    up=194/360
)

sortingFunction0 = pyglitch.InPhasePixelFunction()

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=0,
    reverse=True
)


intervalGenerator1 = pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.LightnessPixelFunction(),
    lo=0.5,
    up=0.747
)

sortingFunction1 = pyglitch.LightnessPixelFunction()

sortFilter1 = pyglitch.PixelSortFilter(
    intervalGenerator1,
    sortingFunction1,
    axis=1,
    reverse=False
)

logging.getLogger().setLevel(7)

I = sortFilter0.apply(_Image.open('flowers.jpg'))

sortFilter1.apply(I).save('flowers_0.jpg')
