# Set working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# vault

from PIL import Image as _Image
import pyglitch
import logging

intervalGenerator0 = pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.HuePixelFunction(),
)

logging.getLogger().setLevel(8)

intervalGenerator0.optimizeImage(_Image.open('vault.jpg'), 37/360, goal=0.7, maxError=0.05)

sortingFunction0 = pyglitch.LightnessPixelFunction()

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=0,
    reverse=True
)

sortFilter0.apply(_Image.open('vault.jpg'), mask=_Image.open('vault_mask.png')).save('vault_0.png')
