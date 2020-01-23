from abc import ABC as _ABC
import random as _random

# Section/Interval Operations


def siOr(s0, s1):
    '''Performs s0 | s1 where s0 and s1 are lists of sections or intervals'''
    actSec = [False, False]
    secs = []

    k0 = [i for s in s0 for i in s]
    k1 = [i for s in s1 for i in s]

    keyPoints = list(sorted([(k, 0) for k in k0]
                          + [(k, 1) for k in k1], key=lambda x: x[0]))

    X = None

    for k, i in keyPoints:
        a0 = actSec[0] | actSec[1]

        actSec[i] = not actSec[i]

        a1 = actSec[0] | actSec[1]

        if a0 != a1:
            if a1:
                X = k
            else:
                if len(secs) > 0 and secs[-1][1] == X:
                    secs[-1] = (secs[-1][0], k)
                else:
                    secs.append((X, k))

    secs = [(a, b) for a, b in secs if a != b]
    return secs


def siAnd(s0, s1):
    '''Performs s0 & s1 where s0 and s1 are lists of sections or intervals'''
    actSec = [False, False]
    secs = []

    k0 = [i for s in s0 for i in s]
    k1 = [i for s in s1 for i in s]

    keyPoints = list(sorted([(k, 0) for k in k0] + [(k, 1)
                                                    for k in k1], key=lambda x: x[0]))

    X = None

    for k, i in keyPoints:
        a0 = actSec[0] & actSec[1]

        actSec[i] = not actSec[i]

        a1 = actSec[0] & actSec[1]

        if a0 != a1:
            if a1:
                X = k
            else:
                if len(secs) > 0 and secs[-1][1] == X:
                    secs[-1] = (secs[-1][0], k)
                else:
                    secs.append((X, k))

    secs = [(a, b) for a, b in secs if a != b]
    return secs


def siXor(s0, s1):
    '''Performs s0 ^ s1 where s0 and s1 are lists of sections or intervals'''
    actSec = [False, False]
    secs = []

    k0 = [i for s in s0 for i in s]
    k1 = [i for s in s1 for i in s]

    keyPoints = list(sorted([(k, 0) for k in k0] + [(k, 1)
                                                    for k in k1], key=lambda x: x[0]))

    X = None

    for k, i in keyPoints:
        a0 = actSec[0] ^ actSec[1]

        actSec[i] = not actSec[i]

        a1 = actSec[0] ^ actSec[1]

        if a0 != a1:
            if a1:
                X = k
            else:
                if len(secs) > 0 and secs[-1][1] == X:
                    secs[-1] = (secs[-1][0], k)
                else:
                    secs.append((X, k))

    secs = [(a, b) for a, b in secs if a != b]
    return secs


def siMinus(s0, s1):
    '''Performs s0 - s1 where s0 and s1 are lists of sections or intervals'''
    actSec = [False, False]
    secs = []

    k0 = [i for s in s0 for i in s]
    k1 = [i for s in s1 for i in s]

    keyPoints = list(sorted([(k, 0) for k in k0] + [(k, 1)
                                                    for k in k1], key=lambda x: x[0]))

    X = None

    active = False

    for k, i in keyPoints:
        actSec[i] = not actSec[i]

        if actSec[0] and not actSec[1]:
            X = k
            active = True
        elif active and X is not None and (not actSec[0] or actSec[1]):
            if len(secs) > 0 and secs[-1][1] == X:
                secs[-1] = (secs[-1][0], k)
            else:
                secs.append((X, k))
            active = False

    secs = [(a, b) for a, b in secs if a != b]

    return secs


class SectionGenerator(_ABC):
    def __init__(self, seed=None):
        self.seed = seed

    def generate(self, xRange, *, _doSeed=True):
        if _doSeed:
            _random.seed(self.seed)
        return []

    def __add__(self, O):
        return OrSectionGenerator(self, O)

    def __or__(self, O):
        return OrSectionGenerator(self, O)

    def __and__(self, O):
        return AndSectionGenerator(self, O)

    def __xor__(self, O):
        return XorSectionGenerator(self, O)

    def __sub__(self, O):
        return MinusSectionGenerator(self, O)


class OrSectionGenerator(SectionGenerator, _ABC):
    def __init__(self, sg0, sg1):
        super().__init__(seed=sg0.seed)
        self.sg0 = sg0
        self.sg1 = sg1

    def generate(self, xRange, *, _doSeed=True):
        return siOr(self.sg0.generate(xRange, _doSeed=_doSeed), self.sg1.generate(xRange, _doSeed=_doSeed))


class AndSectionGenerator(SectionGenerator, _ABC):
    def __init__(self, sg0, sg1):
        super().__init__(seed=sg0.seed)
        self.sg0 = sg0
        self.sg1 = sg1

    def generate(self, xRange, *, _doSeed=True):
        return siAnd(self.sg0.generate(xRange, _doSeed=_doSeed), self.sg1.generate(xRange, _doSeed=_doSeed))


class XorSectionGenerator(SectionGenerator, _ABC):
    def __init__(self, sg0, sg1):
        super().__init__(seed=sg0.seed)
        self.sg0 = sg0
        self.sg1 = sg1

    def generate(self, xRange, *, _doSeed=True):
        return siXor(self.sg0.generate(xRange, _doSeed=_doSeed), self.sg1.generate(xRange, _doSeed=_doSeed))


class MinusSectionGenerator(SectionGenerator, _ABC):
    def __init__(self, sg0, sg1):
        super().__init__(seed=sg0.seed)
        self.sg0 = sg0
        self.sg1 = sg1

    def generate(self, xRange, *, _doSeed=True):
        return siMinus(self.sg0.generate(xRange, _doSeed=_doSeed), self.sg1.generate(xRange, _doSeed=_doSeed))


class PixelFunction(_ABC):
    def __init__(self, coeff=1):
        self.coeff = coeff

    def getValue(self, r, g, b):
        return 0

    def process(self, r, g, b):
        return self.getValue(r, g, b) * self.coeff

    def __add__(self, O):
        return SumPixelFunction(self, O)

    def __mul__(self, c):
        self.coeff = c
        return self

    def __neg__(self):
        self.coeff *= -1
        return self


class SumPixelFunction(PixelFunction, _ABC):
    def __init__(self, f0, f1):
        super().__init__()

        self.functions = [f0, f1]

    def getValue(self, r, g, b):
        return sum(f.getValue(r, g, b) for f in self.functions)

    def __add__(self, O):
        self.functions.append(O)
        return self


class IntervalGenerator(_ABC):
    def __init__(self, seed=None):
        self.seed = seed

    def generate(self, row, *, _doSeed=True):
        if _doSeed:
            _random.seed(self.seed)

        return []

    def __add__(self, O):
        return OrIntervalGenerator(self, O)

    def __or__(self, O):
        return OrIntervalGenerator(self, O)

    def __and__(self, O):
        return AndIntervalGenerator(self, O)

    def __xor__(self, O):
        return XorIntervalGenerator(self, O)

    def __sub__(self, O):
        return MinusIntervalGenerator(self, O)


class OrIntervalGenerator(IntervalGenerator, _ABC):
    def __init__(self, ig0, ig1):
        super().__init__(seed=ig0.seed)
        self.ig0 = ig0
        self.ig1 = ig1

    def generate(self, row, *, _doSeed=True):
        return siOr(self.ig0.generate(row, _doSeed=_doSeed), self.ig1.generate(row, _doSeed=_doSeed))


class AndIntervalGenerator(IntervalGenerator, _ABC):
    def __init__(self, ig0, ig1):
        super().__init__(seed=ig0.seed)
        self.ig0 = ig0
        self.ig1 = ig1

    def generate(self, row, *, _doSeed=True):
        return siAnd(self.ig0.generate(row, _doSeed=_doSeed), self.ig1.generate(row, _doSeed=_doSeed))


class XorIntervalGenerator(IntervalGenerator, _ABC):
    def __init__(self, ig0, ig1):
        super().__init__(seed=ig0.seed)
        self.ig0 = ig0
        self.ig1 = ig1

    def generate(self, row, *, _doSeed=True):
        return siXor(self.ig0.generate(row, _doSeed=_doSeed), self.ig1.generate(row, _doSeed=_doSeed))


class MinusIntervalGenerator(IntervalGenerator, _ABC):
    def __init__(self, ig0, ig1):
        super().__init__(seed=ig0.seed)
        self.ig0 = ig0
        self.ig1 = ig1

    def generate(self, row, *, _doSeed=True):
        return siMinus(self.ig0.generate(row, _doSeed=_doSeed), self.ig1.generate(row, _doSeed=_doSeed))


class ImageFilter(_ABC):
    def __init__(self, seed=None):
        self.seed = seed

    def apply(self, image, verbose=False, *, _doSeed=True):
        if _doSeed:
            _random.seed(self.seed)
        return image.copy()

    def applyChannelwise(self, image, red=True, green=True, blue=True, alpha=False, *, _doSeed=True):
        r, g, b, a = image.convert('RGBA').split()

        if _doSeed:
            _random.seed(self.seed)

        if red:
            logging.log(9, f'Apply imageFilter {repr(self)} to red channel')
            r = self.apply(r.convert('RGBA'), _doSeed=False).convert('L')
        if green:
            logging.log(9, f'Apply imageFilter {repr(self)} to green channel')
            g = self.apply(g.convert('RGBA'), _doSeed=False).convert('L')
        if blue:
            logging.log(9, f'Apply imageFilter {repr(self)} to blue channel')
            b = self.apply(b.convert('RGBA'), _doSeed=False).convert('L')
        if alpha:
            logging.log(9, f'Apply imageFilter {repr(self)} to alpha channel')
            a = self.apply(a.convert('RGBA'), _doSeed=False).convert('L')

        return Image.merge('RGBA', (r, g, b, a))
