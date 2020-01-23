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
    lo=0.85,
    up=0.15
) & pyglitch.PixelFunctionIntervalGenerator(
    pyglitch.LightnessPixelFunction(),
    lo=0.4,
    up=1
)

intervalGenerator0 = pyglitch.ExpandingIntervalGenerator(
    intervalGenerator0, 50) + pyglitch.WaveIntervalGenerator(0.13, percent=True)

sortingFunction0 = pyglitch.LightnessPixelFunction()

sortFilter0 = pyglitch.PixelSortFilter(
    intervalGenerator0,
    sortingFunction0,
    axis=1
)

logging.getLogger().setLevel(7)

I = sortFilter0.apply(_Image.open('0018.png')).save('0018_0.png')
