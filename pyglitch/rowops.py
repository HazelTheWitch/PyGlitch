import random as _random
import numpy as _np
from math import exp as _exp

from PIL import Image as _Image

import logging as _logging

from colorsys import rgb_to_hsv as _rgb_to_hsv, rgb_to_hls as _rgb_to_hls

from .glitchabc import siOr, siAnd, siXor, siMinus, SectionGenerator, OrSectionGenerator, AndSectionGenerator, XorSectionGenerator, MinusSectionGenerator, PixelFunction, SumPixelFunction, IntervalGenerator, OrIntervalGenerator, AndIntervalGenerator, XorIntervalGenerator, MinusIntervalGenerator, ImageFilter

# Probability Functions
def _uniformProb(maxY):
    def uniform(y):
        return _random.random() < 0.5
    return uniform


def _sigmoidProb(maxY):
    invMaxY = 1 / maxY

    def sigmoidProb(y):
        return _random.random() < 1 / (1 + exp(-y * invMaxY))
    return sigmoidProb

# Max Functions
def _hardMax(maxY):
    def hardMax(y):
        if y < -maxY:
            y = -maxY
        if y > maxY:
            y = maxY

        return y
    return hardMax


def _noMax(maxY):
    def noMax(y):
        return y
    return noMax


def _wrappingMax(maxY):
    def wrappingMax(y):
        if y < -maxY:
            y = maxY
        if y > maxY:
            y = -maxY

        return y
    return wrappingMax


PROB_FUNCS = {'uniform': _uniformProb, 'sigmoid': _sigmoidProb}
MAX_FUNCS = {'hard': _hardMax, 'no': _noMax, 'wrapping': _wrappingMax}


class FullSectionGenerator(SectionGenerator):
    def generate(self, xRange, *, _doSeed=True):
        super().generate(xRange, _doSeed=_doSeed)

        return [(0, xRange)]


class RandomWalkSectionGenerator(SectionGenerator):
    '''Forms sections '''

    def __init__(self, seed=None, maxY=10, step=1, y0=0, maxFunc='hard', probFunc='uniform', cutOff=None):
        super().__init__(seed)

        if cutOff is None:
            cutOff = y0
        if not (0 < maxY < 128):
            raise ValueError(f'`maxY` must be in range (0, 128) not {maxY}')

        try:
            probFunc = PROB_FUNCS[probFunc](maxY)
        except KeyError:
            raise ValueError(
                f'`probFunc` must be a valid key of PROB_FUNCS not {probFunc}')

        try:
            maxFunc = MAX_FUNCS[maxFunc](maxY)
        except KeyError:
            raise ValueError(
                f'`maxFunc` must be a valid key of MAX_FUNCS not {maxFunc}')

        self.maxY = maxY
        self.step = step
        self.y0 = y0
        self.maxFunc = maxFunc
        self.probFunc = probFunc
        self.cutOff = cutOff

    def generate(self, xRange, *, _doSeed=True):
        sections = super().generate(xRange, _doSeed=_doSeed)

        y = _np.zeros((xRange,))

        y0 = self.maxFunc(self.y0)

        y[0] = y0
        Y = y0

        for i in range(1, xRange):
            if self.probFunc(Y):
                Y -= self.step
            else:
                Y += self.step

            Y = self.maxFunc(Y)

            y[i] = Y

        X = 0
        inSection = y[0] > self.cutOff

        for i in range(1, xRange):
            aboveCutoff = y[i] > self.cutOff
            if inSection:
                if not aboveCutoff:
                    sections.append((X, i))
                    inSection = False
            else:
                if aboveCutoff:
                    X = i
                    inSection = True

        if inSection:
            sections.append((X, xRange))

        return sections


class UniformSectionGenerator(SectionGenerator):
    '''Forms sections uniformly, takes longer the closer `sectionCount` gets to `xRange // 2` due to random chance of overlap'''

    def __init__(self, seed=None, sectionCount=10):
        super().__init__(seed)

        self.sectionCount = sectionCount

    def generate(self, xRange, *, _doSeed=True):
        sections = super().generate(xRange, _doSeed=_doSeed)

        if not (0 < self.sectionCount < xRange // 2):
            raise ValueError(
                f'`sectionCount` must be greater than 0 and less than `xRange // 2` not {sectionCount}')

        xS = set()

        while len(xS) < self.sectionCount * 2:
            x = _random.randrange(xRange + 1)

            if x not in xS:
                xS.add(x)

        xS = list(sorted(xS))

        for i in range(self.sectionCount):
            I = 2*i
            sections.append((xS[I], xS[I + 1]))

        return sections


class RandomPixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return _random.random()


class MaxRGBPixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return max(r, g, b)/256


class HuePixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return rgb_to_hsv(r/256, g/256, b/256)[0]


class SaturationPixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return rgb_to_hsv(r/256, g/256, b/256)[1]


class ValuePixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return rgb_to_hsv(r/256, g/256, b/256)[2]


class LightnessPixelFunction(PixelFunction):
    def getValue(self, r, g, b):
        return rgb_to_hls(r/256, g/256, b/256)[2]


class FullIntervalGenerator(IntervalGenerator):
    def generate(self, row, *, _doSeed=True):
        super().generate(row, _doSeed=_doSeed)

        L = row.shape[0]

        return [(0, L)]


class WaveIntervalGenerator(IntervalGenerator):
    def __init__(self, width, percent=False, minRand=0.95, maxRand=1.05, seed=None):
        super().__init__(seed=seed)
        self.width = width
        self.percent = percent

        self.minRand = minRand
        self.maxRand = maxRand

    def generate(self, row, *, _doSeed=True):
        super().generate(row, _doSeed=_doSeed)

        L = row.shape[0]

        if self.percent:
            width = int(L * self.width)
        else:
            width = self.width

        i0 = _random.randrange(width)

        if self.minRand == self.maxRand:
            return [(i, min(L, int(i+width))) for i in range(i0, L, 2 * width)]
        else:
            return [(i, min(L, int(i+_random.uniform(width*0.95, min(width*1.05, i + 2 * width - 1))))) for i in range(i0, L, 2 * width)]


class SectionBasedIntervalGenerator(IntervalGenerator):
    def __init__(self, sectionGenerator, seed=None):
        super().__init__(seed=seed)
        self.sectionGenerator = sectionGenerator

    def generate(self, row, *, _doSeed=True):
        super().generate(row, _doSeed=_doSeed)

        L = row.shape[0]

        return self.sectionGenerator.generate(L)


class UniformIntervalGenerator(SectionBasedIntervalGenerator):
    def __init__(self, nIntervals, seed=None):
        super().__init__(UniformSectionGenerator(
            seed=seed, sectionCount=nIntervals), seed=seed)


class PixelFunctionIntervalGenerator(IntervalGenerator):
    def __init__(self, pixelFunction, lo=0.2, up=0.8, seed=None):
        super().__init__(seed=seed)
        self.pixelFunction = pixelFunction
        self.lo = lo
        self.up = up

    def inRange(self, v):
        if self.lo < self.up:
            return self.lo < v < self.up
        else:
            return not (self.up < v < self.lo)

    def generate(self, row, *, _doSeed=True):
        super().generate(row, _doSeed=_doSeed)

        L = row.shape[0]

        intervals = []

        X = 0
        inSection = self.inRange(self.pixelFunction.process(*row[0, :]))

        for i in range(1, L):
            aboveCutoff = self.inRange(self.pixelFunction.process(*row[i, :]))
            if inSection:
                if not aboveCutoff:
                    intervals.append((X, i))
                    inSection = False
            else:
                if aboveCutoff:
                    X = i
                    inSection = True

        if inSection:
            intervals.append((X, L))

        return intervals


class CappingIntervalGenerator(IntervalGenerator):
    def __init__(self, intervalGenerator, maxIntervalSize, percent=False):
        super().__init__(seed=intervalGenerator.seed)

        self.maxIntervalSize = maxIntervalSize
        self.intervalGenerator = intervalGenerator

        self.percent = percent

    def generate(self, row, *, _doSeed=True):
        intervals = self.intervalGenerator.generate(row, _doSeed=_doSeed)
        newIntervals = []

        maxIntervalSize = self.maxIntervalSize

        if self.percent:
            L = row.shape[0]
            maxIntervalSize *= L

            maxIntervalSize = int(maxIntervalSize)

        for x0, x1 in intervals:
            while x1 - x0 > maxIntervalSize:
                newIntervals.append((x0, x0+maxIntervalSize))
                x0 += maxIntervalSize
            newIntervals.append((x0, x1))

        return newIntervals


class ShiftFilter(ImageFilter):
    def __init__(self, sectionGenerator, shiftFunction=10, axis=0, seed=None, shiftSame=True):
        super().__init__(seed=seed)

        if type(shiftFunction) == int:
            shiftFunction = ShiftFilter.constantShift(shiftFunction)

        self.sectionGenerator = sectionGenerator

        self.shiftFunction = shiftFunction

        if axis not in (0, 1):
            raise ValueError(f'`axis` must be 0 or 1 not {axis}')

        self.axis = axis

        if shiftSame == True:
            shiftSame = 1 - self.axis
        elif shiftSame == False:
            shiftSame = self.axis

        self.shiftAxis = shiftSame

    def apply(self, image, verbose=False, *, _doSeed=True):
        image = super().apply(image, _doSeed=_doSeed)

        A = _np.array(image)

        xRange = A.shape[self.axis]

        sections = self.sectionGenerator.generate(xRange)

        logging.log(8, f'{len(sections)} sections found')

        for s in sections:
            x0, x1 = s

            I = [slice(L) for L in A.shape]

            I[self.axis] = slice(x0, x1)

            I = tuple(I)

            sAmount = self.shiftFunction(s)

            logging.log(8, f'Shifting section {s} {sAmount} pixels')

            A[I] = _np.roll(A[I], sAmount, axis=self.shiftAxis)

        return Image.fromarray(A)

    @staticmethod
    def constantShift(delta):
        def shift(section):
            return delta
        return shift

    @staticmethod
    def uniformShift(minDelta, maxDelta):
        def shift(section):
            return _random.randint(minDelta, maxDelta)
        return shift


class PixelSortFilter(ImageFilter):
    def __init__(self, intervalGenerator, sortingFunction, sectionGenerator=None, seed=None, axis=0, reverse=False):
        super().__init__(seed=seed)

        if sectionGenerator is None:
            sectionGenerator = FullSectionGenerator(seed=seed)

        self.sectionGenerator = sectionGenerator
        self.intervalGenerator = intervalGenerator

        self.sortingFunction = sortingFunction

        self.axis = axis

        self.sortAxis = 1 - self.axis

        self.reverse = reverse

    def apply(self, image, mask=None, mCutOff=128, maskAfter=False, angle=0, *, _doSeed=True):
        image = super().apply(image, _doSeed=_doSeed)

        angle = -angle

        oSize = list(reversed(image.size))

        if angle != 0:
            image = image.rotate(angle, expand=True)

        A = _np.array(image)
        A0 = None

        if angle != 0:
            if mask is None:
                M = _np.ones(oSize, dtype=_np.uint8) * 255
                mask = Image.fromarray(M)

            mask = mask.rotate(angle, expand=True)

        if mask is not None:
            M = _np.array(mask.convert('L')) > mCutOff
            if not maskAfter:
                M = _np.stack([M, M, M], axis=2)
                M = M.astype(_np.uint8) * 255

                pfig = PixelFunctionIntervalGenerator(
                    MaxRGBPixelFunction(), -1, 0.5)

            A0 = _np.copy(A)

        xRange = A.shape[self.axis]

        sections = self.sectionGenerator.generate(xRange)

        logging.log(9, f'Found {len(sections)} sections')

        pixelsSorted = 0
        intervalsSorted = 0

        for s in sections:
            x0, x1 = s

            I0 = [slice(L) for L in A.shape]

            logging.log(8, f'Sorting section {s}')

            for i in range(x0, x1):
                I0[self.axis] = i

                I = tuple(I0)

                intervals = self.intervalGenerator.generate(A[I])

                if mask is not None and not maskAfter:
                    maskIntervals = pfig.generate(M[I], _doSeed=False)

                    intervals = siMinus(intervals, maskIntervals)

                intervalsSorted += len(intervals)

                pToSort = sum(map(lambda i: i[1] - i[0], intervals))
                pixelsSorted += pToSort
                logging.log(
                    7, f'Sorting index {i} in section {s} - {len(intervals)} intervals - {pToSort} pixels to sort')
                I1 = [slice(L) for L in A.shape]
                I1[self.axis] = i

                for i0, i1 in intervals:
                    logging.log(
                        6, f'Sorting interval {(i0, i1)} in section {s}')
                    I1[self.sortAxis] = slice(i0, i1)

                    I = tuple(I1)

                    A[I] = sorted(A[I], key=lambda p: self.sortingFunction.process(
                        *p), reverse=self.reverse)

        if mask is not None and maskAfter:
            logging.log(9, f'Applying Mask')
            A[M] = A0[M]

        logging.log(
            9, f'Sorted {pixelsSorted} pixels in {intervalsSorted} intervals across {len(sections)} sections')

        image = Image.fromarray(A)

        if angle != 0:
            oSize = list(reversed(oSize))
            image = image.rotate(-angle)
            nSize = image.size

            left = (nSize[0] - oSize[0]) // 2
            upper = (nSize[1] - oSize[1]) // 2
            right = (nSize[0] + oSize[0]) // 2
            lower = (nSize[1] + oSize[1]) // 2

            image = image.crop(box=(left, upper, right, lower))
        return image


# Helper Functions
def loUpAround(v, dev, up=None, inverse=False):
    if up is None:
        up = 1 + dev
        lo = 1 - dev
    else:
        lo = dev

    if not inverse:
        return {'lo': v * lo, 'up': up}
    else:
        return {'up': v * lo, 'lo': up}
