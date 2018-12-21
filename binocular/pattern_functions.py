"""Definition of patterns to be learned.

Each pattern defines a univariate timeseries. They are represented by functions that map form
the natural numbers (time step) to a real value."""
import numpy as np

np.random.seed(0)
# The following patterns are created.
# 1 = sine10  2 = sine15  3 = sine20  4 = spike20
# 5 = spike10 6 = spike7  7 = 0   8 = 1
# 9 = rand4; 10 = rand5  11 = rand6 12 = rand7
# 13 = rand8 14 = sine10range01 15 = sine10rangept5pt9
# 16 = rand3 17 = rand9 18 = rand10 19 = 0.8 20 = sineroot27
# 21 = sineroot19 22 = sineroot50 23 = sineroot75
# 24 = sineroot10 25 = sineroot110 26 = sineroot75tenth
# 27 = sineroots20plus40  28 = sineroot75third
# 29 = sineroot243  30 = sineroot150  31 = sineroot200
# 32 = sine10.587352723 33 = sine10.387352723
# 34 = rand7  35 = sine12  36 = 10+perturb  37 = sine11
# 38 = sine10.17352723  39 = sine5 40 = sine6
# 41 = sine7 42 = sine8  43 = sine9 44 = sine12
# 45 = sine13  46 = sine14  47 = sine10.8342522
# 48 = sine11.8342522  49 = sine12.8342522  50 = sine13.1900453
# 51 = sine7.1900453  52 = sine7.8342522  53 = sine8.8342522
# 54 = sine9.8342522 55 = sine5.19004  56 = sine5.8045
# 57 = sine6.49004 58 = sine6.9004 59 = sine13.9004
# 60 = 18+perturb  61 = spike3  62 = spike4 63 = spike5
# 64 = spike6 65 = rand4  66 = rand5  67 = rand6 68 = rand7
# 69 = rand8 70 = rand4  71 = rand5  72 = rand6 73 = rand7
# 74 = rand8

patterns = []
for i in range(78):
    patterns.append(1)

patterns[0] = lambda t: np.sin(2 * np.pi * t / 10)
patterns[1] = lambda t: np.sin(2 * np.pi * t / 10)
patterns[2] = lambda t: np.sin(2 * np.pi * t / 15)
patterns[3] = lambda t: np.sin(2 * np.pi * t / 20)
patterns[4] = lambda t: +(1 == np.mod(t, 20))
patterns[5] = lambda t: +(1 == np.mod(t, 10))
patterns[6] = lambda t: +(1 == np.mod(t, 7))
patterns[7] = lambda t: 0
patterns[8] = lambda t: 1

rp = np.random.randn(4)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[9] = lambda t: rp[np.mod(t, 4)]

rp10 = np.random.rand(5)
maxVal = max(rp10)
minVal = min(rp10)
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9
patterns[10] = lambda t: (rp10[np.mod(t, 5)])

rp = np.random.rand(6)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[11] = lambda t: (rp[np.mod(t, 6)])

rp = np.random.rand(7)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[12] = lambda t: (rp[np.mod(t, 7)])

rp = np.random.rand(8)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[13] = lambda t: (rp[np.mod(t, 8)])

patterns[14] = lambda t: 0.5 * np.sin(2 * np.pi * t / 10) + 0.5

patterns[15] = lambda t: 0.2 * np.sin(2 * np.pi * t / 10) + 0.7

rp = np.random.randn(3)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[16] = lambda t: rp[np.mod(t, 3)]

rp = np.random.randn(9)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[17] = lambda t: rp[np.mod(t, 9)]

rp18 = np.random.randn(10)
maxVal = max(rp18)
minVal = min(rp18)
rp18 = 1.8 * (rp18 - minVal) / (maxVal - minVal) - 0.9
patterns[18] = lambda t: rp18[np.mod(t, 10)]

patterns[19] = lambda t: 0.8
patterns[20] = lambda t: np.sin(2 * np.pi * t / np.sqrt(27))
patterns[21] = lambda t: np.sin(2 * np.pi * t / np.sqrt(19))
patterns[22] = lambda t: np.sin(2 * np.pi * t / np.sqrt(50))
patterns[23] = lambda t: np.sin(2 * np.pi * t / np.sqrt(75))
patterns[24] = lambda t: np.sin(2 * np.pi * t / np.sqrt(10))
patterns[25] = lambda t: np.sin(2 * np.pi * t / np.sqrt(110))
patterns[26] = lambda t: 0.1 * np.sin(2 * np.pi * t / np.sqrt(75))
patterns[27] = lambda t: 0.5 * (np.sin(2 * np.pi * t / np.sqrt(20)) + np.sin(2 * np.pi * t / np.sqrt(40)))
patterns[28] = lambda t: 0.33 * np.sin(2 * np.pi * t / np.sqrt(75))
patterns[29] = lambda t: np.sin(2 * np.pi * t / np.sqrt(243))
patterns[30] = lambda t: np.sin(2 * np.pi * t / np.sqrt(150))
patterns[31] = lambda t: np.sin(2 * np.pi * t / np.sqrt(200))
patterns[32] = lambda t: np.sin(2 * np.pi * t / 10.587352723)
patterns[33] = lambda t: np.sin(2 * np.pi * t / 10.387352723)

rp = np.random.rand(7)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[34] = lambda t: (rp[np.mod(t, 7)])

patterns[35] = lambda t: np.sin(2 * np.pi * t / 12)

rpDiff = np.random.randn(5)
rp = rp10 + 0.2 * rpDiff
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[36] = lambda t: rp[np.mod(t, 5)]
patterns[37] = lambda t: np.sin(2 * np.pi * t / 11)
patterns[38] = lambda t: np.sin(2 * np.pi * t / 10.17352723)
patterns[39] = lambda t: np.sin(2 * np.pi * t / 5)
patterns[40] = lambda t: np.sin(2 * np.pi * t / 6)
patterns[41] = lambda t: np.sin(2 * np.pi * t / 7)
patterns[42] = lambda t: np.sin(2 * np.pi * t / 8)
patterns[43] = lambda t: np.sin(2 * np.pi * t / 9)
patterns[44] = lambda t: np.sin(2 * np.pi * t / 12)
patterns[45] = lambda t: np.sin(2 * np.pi * t / 13)
patterns[46] = lambda t: np.sin(2 * np.pi * t / 14)
patterns[47] = lambda t: np.sin(2 * np.pi * t / 10.8342522)
patterns[48] = lambda t: np.sin(2 * np.pi * t / 11.8342522)
patterns[49] = lambda t: np.sin(2 * np.pi * t / 12.8342522)
patterns[50] = lambda t: np.sin(2 * np.pi * t / 13.1900453)
patterns[51] = lambda t: np.sin(2 * np.pi * t / 7.1900453)
patterns[52] = lambda t: np.sin(2 * np.pi * t / 1.8342522)
patterns[53] = lambda t: np.sin(2 * np.pi * t / 4.8342522)
patterns[54] = lambda t: np.sin(2 * np.pi * t / 8.8342522)
patterns[55] = lambda t: np.sin(2 * np.pi * t / 5.1900453)
patterns[56] = lambda t: np.sin(2 * np.pi * t / 5.804531)
patterns[57] = lambda t: np.sin(2 * np.pi * t / 6.4900453)
patterns[58] = lambda t: np.sin(2 * np.pi * t / 6.900453)
patterns[59] = lambda t: np.sin(2 * np.pi * t / 13.900453)

rpDiff = np.random.randn(10)
rp = rp18 + 0.3 * rpDiff
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[60] = lambda t: (rp[np.mod(t, 10)])

patterns[61] = lambda t: +(1 == np.mod(t, 3))
patterns[62] = lambda t: +(1 == np.mod(t, 4))
patterns[63] = lambda t: +(1 == np.mod(t, 5))
patterns[64] = lambda t: +(1 == np.mod(t, 6))

rp = np.random.randn(4)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[65] = lambda t: rp[np.mod(t, 4)]

rp10 = np.random.rand(5)
maxVal = max(rp10)
minVal = min(rp10)
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9
patterns[66] = lambda t: (rp10[np.mod(t, 5)])

rp = np.random.rand(6)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[67] = lambda t: (rp[np.mod(t, 6)])
rp = np.random.rand(7)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[68] = lambda t: (rp[np.mod(t, 7)])

rp = np.random.rand(8)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[69] = lambda t: (rp[np.mod(t, 8)])

rp = np.random.randn(4)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[70] = lambda t: rp[np.mod(t, 4)]

rp10 = np.random.rand(5)
maxVal = max(rp10)
minVal = min(rp10)
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9
patterns[71] = lambda t: (rp10[np.mod(t, 5)])

rp = np.random.rand(6)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[72] = lambda t: (rp[np.mod(t, 6)])

rp = np.random.rand(7)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[73] = lambda t: (rp[np.mod(t, 7)])

rp = np.random.rand(8)
maxVal = rp[np.argmax(rp)]
minVal = rp[np.argmin(rp)]
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9
patterns[74] = lambda t: (rp[np.mod(t, 8)])

patterns[75] = lambda t: 1.0 * np.sin(2 * np.pi * t / 0.04373282)
patterns[76] = lambda t: 1.0 * np.sin(2 * np.pi * t / 0.0721342522)

rp = np.array([0.515225899591, 0.641977677728,
               0.619641652873, 0.9, -0.336278680907])
patterns[77] = lambda t: (rp[np.mod(t, 5)])


def pattern_interface(pattern):
    def vector_pattern(t):
        return np.array([pattern(t)])
    return vector_pattern

for i, pattern in enumerate(patterns):
    patterns[i] = pattern_interface(pattern)
