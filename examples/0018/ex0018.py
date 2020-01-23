# Set working directory
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# 0018

from PIL import Image as _Image
import pyglitch
import logging

intervalGenerator0 = pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.HuePixelFunction(),
    lo=0.9,
    up=0.1
) & pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.LightnessPixelFunction(),
    lo=0.4,
    up=1
)

intervalGenerator0 = pyglitch.ExpandingIntervalGenerator(
    intervalGenerator0, 50) + pyglitch.WaveIntervalGenerator(0.04, percent=True)

sortingFunction0 = pyglitch.SaturationPixelFunction()

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=0,
    reverse=True
)

logging.getLogger().setLevel(7)

I = sortFilter0.apply(_Image.open('0018.png')).save('0018_0.png')
