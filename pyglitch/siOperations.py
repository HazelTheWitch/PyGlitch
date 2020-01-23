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


def verifySI(s):
    keyPoints = []

    for x0, x1 in s:
        keyPoints.append((x0, True))
        keyPoints.append((x1, False))
    
    keyPoints.sort(key=lambda x: x[0])

    newS = []
    X = None
    depth = 0

    for x, inS in keyPoints:
        if inS:
            if depth == 0:
                X = x
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                newS.append((X, x))
         
    return newS
