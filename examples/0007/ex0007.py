# Set working directory 
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 0007
import pyglitch
from PIL import Image
import logging


intervalGenerator = pyglitch.PixelFunctionIntervalGenerator( # Intervals based on PixelFunction
    pyglitch.HuePixelFunction(), # Returns hue value for given pixel [0,1]
    lo=0.9, # Low value for interval definition, all pixels in intervals are in (lo, up) unless lo > up
    up=0.1  # Then all pixels are not in (up, lo)
)

 # Pixels are sorted within intervals by the sorting function, PixelFunctions can be easily added/multiplied
 # PixelFunctions are not recreated when operated on
sortingFunction = pyglitch.SaturationPixelFunction()

sortFilter = pyglitch.PixelSortFilter(intervalGenerator, sortingFunction, axis=1, reverse=True)

logging.getLogger().setLevel(7) # Set logging level to view progress

sortFilter.apply(Image.open('0007.png')).save('0007_0.png')
