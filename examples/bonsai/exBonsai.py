# Set working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# bonsai

from PIL import Image as _Image
import pyglitch
import logging

intervalGenerator0 = pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.HuePixelFunction(),
)

logging.getLogger().setLevel(7)

intervalGenerator0.optimizeImage(_Image.open('bonsai.jpg'), 333/360, goal=0.45, iterations=3)

sortingFunction0 = pyglitch.LightnessPixelFunction()

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=1
)

sortFilter0.apply(_Image.open('bonsai.jpg'), angle=45).save('bonsai_0.png')
