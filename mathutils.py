def getIntegerPlaces(theNumber):
    import math
    if theNumber <= 999999999999997:
        return int(math.log10(theNumber)) + 1
    else:
        counter = 15
        while theNumber >= 10**counter:
            counter += 1
        return counter
    