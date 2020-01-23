# Set working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# salt

import logging
from PIL import Image
import pyglitch

intervalGenerator = pyglitch.PixelFunctionIntervalGenerator(  # Intervals based on PixelFunction
    pyglitch.LightnessPixelFunction(),  # Returns hue value for given pixel [0,1]
    # Low value for interval definition, all pixels in intervals are in (lo, up) unless lo > up
    lo=0.0,
    up=0.7
)

sortingFunction = pyglitch.LightnessPixelFunction()

# Axis 0 for horizontal
sortFilter = pyglitch.PixelSortFilter(
    intervalGenerator, sortingFunction, axis=0, reverse=True)

logging.getLogger().setLevel(7)  # Set logging level to view progress

sortFilter.apply(Image.open('salt.jpg')).save('salt_0.jpg')
