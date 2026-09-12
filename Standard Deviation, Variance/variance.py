from math import sqrt
from statistics import stdev, variance


speed = [32,111,138,28,59,77,97]

print(variance(speed))
print(sqrt(variance(speed))) # [STANDARD DEVIATION]
print(stdev(speed))


# [CALCULATION OF VARIANCE]
mean = (32+111+138+28+59+77+97)/7
print(mean)
difference = [speed - mean for speed in speed]
print(difference)
squared = [d**2 for d in difference]
print(squared)
variance = sum(squared) / len(squared)
print(variance)

standard_deviation = sqrt(variance)
print(standard_deviation)