# roger truong
# MAE 157


## last tested with Python 3.9.9


import pandas
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal


## INIT CONDITIONS -----------------------------------------------------------------------

correctFactor = 0.75 # for stainless steel
# https://www.omega.com/en-us/control-monitoring/motion-and-position/displacement-transducers/ld701/p/LD701-5-10

senseStartmm = 5 # mm
senseEndmm = 10 # mm
senseStartV = 1 # V
senseEndV = 9 # V
convFactor = (senseEndmm - senseStartmm) / (senseEndV - senseStartV)
timeStep = 2e-3 # 2 milliseconds


massWheel = 1.03 # kg
lengthShaft = ((19+(11/16))-(3+(5/8)))*0.0254 # inches to meters
diaShaft = 6e-3 # m
youngMod = 193e9 # GPa to Pa
# assumed stainless steel 304
# E from "Materials Science and Engineering: An Introduction 9th Edition (Callister)"
inertia = np.pi/4 * np.power(diaShaft/2, 4)
# https://www.engineeringtoolbox.com/area-moment-inertia-d_1328.html


# plot stuff
dotSize = 5
f = 1 # increment the figure number each plot

## FUNCTIONS -----------------------------------------------------------------------------


def voltTomm(inputVolt):
    outputX = ((inputVolt - senseStartV) * convFactor) + senseStartmm
    actualOutputX = outputX * correctFactor
    return actualOutputX
# uses conversion factor and correction factor to interpolate voltage to displacement

def plotScatter(xaxis, column, color="", xcolor="black"):
    tmpData = dispInmmCut[column]
    if color=="":
        plt.scatter(xaxis, tmpData, s=dotSize, label=column)
    else:
        plt.scatter(xaxis, tmpData, s=dotSize, label=column, color=color)
    locMaxI, _ = scipy.signal.find_peaks(tmpData)
    plt.scatter(xaxis[locMaxI], tmpData.iloc[locMaxI], marker="x", color=xcolor, s=50)
    saveArray1 = pandas.DataFrame(locMaxI*2e-3, columns=[str(column) + " time"])
    saveArray2 = pandas.DataFrame(tmpData.iloc[locMaxI].reset_index(drop=True))\
        .rename(columns={column:(str(column)+" maxima")})
    return saveArray1, saveArray2
# plots the given dataset(s) as a scatter plot, and finds the local maxima, marked with a black x

# color reference:
# https://matplotlib.org/stable/_images/sphx_glr_named_colors_003_2_0x.png



## PLOTTING -------------------------------------------------------------------------------------

df = pandas.read_csv("data_raw_volts.csv")
dispInmm = voltTomm(df)
dispInmm = dispInmm.fillna(0)
# read raw data and convert it to mm

timeLength = len(dispInmm)
timeArray = np.arange(0, timeStep*timeLength, timeStep)
# initialize the corresponding time array

colNames = list(dispInmm.columns.values)
FRNames = list(["H1", "H2", "H3"])
# list of names for potential use

maxArray = pandas.DataFrame()
eccArray1 = pandas.DataFrame()
eccArray2 = pandas.DataFrame()
eccArray3 = pandas.DataFrame()
# initalize a bunch of empty arrays


# plotting everything
startTime = 1.25 # seconds
endTime = 2 # seconds

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
for i in range(0, len(colNames)):
    plotScatter(timeArrayCut, colNames[i])
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# Hammer free response 1
startTime = 1.25 # seconds
endTime = 2.5 # seconds

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "H1", "darkturquoise")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# Hammer free response 2
startTime = 0.85 # seconds
endTime = 1.25 # seconds

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "H2", "darkorange")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# Hammer free response 3
startTime = 0.85 # seconds
endTime = 1.15 # seconds

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "H3", "forestgreen")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# resonance 1
startTime = 0.5 # seconds
endTime = 3.5 # seconds
# endTime = 0.75

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "R1", "deepskyblue")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# resonance 2
startTime = 0.5 # seconds
endTime = 5.5 # seconds
# endTime = 0.75

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "R2", "goldenrod")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1


# resonance 3
startTime = 0.5 # seconds
endTime = 6.5 # seconds
# endTime = 0.75

dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
plt.figure(f)
tmpArray1, tmpArray2 = plotScatter(timeArrayCut, "R3", "seagreen")
maxArray = pandas.concat([maxArray, tmpArray1, tmpArray2], axis = 1)
plt.legend(loc="lower right")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s)")
f = f+1




## CALCS ----------------------------------------------------------------------------------------------------

startTime = 0.5 # seconds
endTime = 3 # seconds
dispInmmCut = dispInmm.iloc[int(startTime*1000/2):int(endTime*1000/2)]
timeArrayCut = timeArray[int(startTime*1000/2) : int(endTime*1000/2)]
# reset the start/end time for eccentricity calcs

maxDiffArray = maxArray.diff(periods=1)
# find the delta between values in maxArray


# for H1
# x = L/2
H1Stats = [maxDiffArray["H1 time"].mean(), maxDiffArray["H1 time"].std(), \
    maxDiffArray["H1 maxima"].mean(), maxDiffArray["H1 maxima"].std()]
R1Stats = [maxDiffArray["R1 time"].mean(), maxDiffArray["R1 time"].std(), \
    maxDiffArray["R1 maxima"].mean(), maxDiffArray["R1 maxima"].std()]
# finds the statistical distributions (mean and std)

kDamp = 48*youngMod*inertia / np.power(lengthShaft, 3)
# calculates the spring constant

delta = np.log(maxArray.at[0, "H1 maxima"] / maxArray.at[1, "H1 maxima"])
# calculates logarithmic decrement

zetaExp = delta / np.sqrt(4*np.power(np.pi, 2) - np.power(delta, 2))
# calculates damping ratio

natF = np.sqrt(kDamp/massWheel)
dampF = natF * np.sqrt(1-np.power(zetaExp, 2))
# finds theoretical natural and damped frequency

dampFexp = 2*np.pi / H1Stats[0]
natFexp = dampFexp / np.sqrt(1-np.power(zetaExp, 2))
# finds experimental natural and damped frequency

resF = natF * np.sqrt(1 - 2*np.power(zetaExp, 2))
resFexp = 2*np.pi / R1Stats[0]
# finds experimental/theoretical resonant frequency


lamda = dampFexp / natFexp
BANames = list(["BR1", "BR2", "BR3", "BR4", "BR5", "AR1", "AR2", "AR3", "AR4", "AR5", "R1"])
for i in range(0, len(BANames)):
    ecc = (kDamp * dispInmmCut[BANames[i]] * np.max(dispInmmCut[BANames[i]]) * \
        np.sqrt(np.power(1-np.power(lamda, 2), 2) + np.power(2 * zetaExp * lamda, 2))) \
        / (massWheel * np.power(dampFexp, 2))
    ecc = ecc.dropna()
    eccArray1 = pandas.concat([eccArray1, ecc], axis=1)
# calculates eccentricity for each dataset in this trial


print("H1")
print("k = " + str(kDamp))
print("\u03B4 = " + str(delta))
print("\u03B6 = " + str(zetaExp))
print("theo. \u03C9_n = " + str(natF))
print("theo. \u03C9_d = " + str(dampF))
print("exp. \u03C9_n = " + str(natFexp))
print("exp. \u03C9_d = " + str(dampFexp))
print("")
print("R1")
print("theo. \u03C9_r = " + str(resF))
print("exp. \u03C9_r = " + str(resFexp))
print("")


# for H2
# x = L/4
H2Stats = [maxDiffArray["H2 time"].mean(), maxDiffArray["H2 time"].std(), \
    maxDiffArray["H2 maxima"].mean(), maxDiffArray["H2 maxima"].std()]
R2Stats = [maxDiffArray["R2 time"].mean(), maxDiffArray["R2 time"].std(), \
    maxDiffArray["R2 maxima"].mean(), maxDiffArray["R2 maxima"].std()]
# finds the statistical distributions (mean and std)

kDamp = 256*youngMod*inertia / (3*np.power(lengthShaft, 3))
# calculates the spring constant

delta = np.log(maxArray.at[0, "H2 maxima"] / maxArray.at[1, "H2 maxima"])
# calculates logarithmic decrement

zetaExp = delta / np.sqrt(4*np.power(np.pi, 2) - np.power(delta, 2))
# calculates damping ratio

natF = np.sqrt(kDamp/massWheel)
dampF = natF * np.sqrt(1-np.power(zetaExp, 2))
# finds theoretical natural and damped frequency

dampFexp = 2*np.pi / H2Stats[0]
natFexp = dampFexp / np.sqrt(1-np.power(zetaExp, 2))
# finds experimental natural and damped frequency

resF = natF * np.sqrt(1 - 2*np.power(zetaExp, 2))
resFexp = 2*np.pi / R2Stats[0]
# finds experimental/theoretical resonant frequency

lamda = dampFexp / natFexp
BANames = list(["BR6", "BR7", "BR8", "BR9", "BR10", "AR6", "AR7", "AR8", "AR9", "AR10", "R2"])
for i in range(0, len(BANames)):
    ecc = (kDamp * dispInmmCut[BANames[i]] * np.max(dispInmmCut[BANames[i]]) * \
        np.sqrt(np.power(1-np.power(lamda, 2), 2) + np.power(2 * zetaExp * lamda, 2))) \
        / (massWheel * np.power(dampFexp, 2))
    ecc = ecc.dropna()
    eccArray2 = pandas.concat([eccArray2, ecc], axis=1)
# calculates eccentricity for each dataset in this trial


print("H2")
print("k = " + str(kDamp))
print("\u03B4 = " + str(delta))
print("\u03B6 = " + str(zetaExp))
print("theo. \u03C9_n = " + str(natF))
print("theo. \u03C9_d = " + str(dampF))
print("exp. \u03C9_n = " + str(natFexp))
print("exp. \u03C9_d = " + str(dampFexp))
print("")
print("R2")
print("theo. \u03C9_r = " + str(resF))
print("exp. \u03C9_r = " + str(resFexp))
print("")


# for H3
# x = 0.14786L
H3Stats = [maxDiffArray["H3 time"].mean(), maxDiffArray["H3 time"].std(), \
    maxDiffArray["H3 maxima"].mean(), maxDiffArray["H3 maxima"].std()]
R3Stats = [maxDiffArray["R3 time"].mean(), maxDiffArray["R3 time"].std(), \
    maxDiffArray["R3 maxima"].mean(), maxDiffArray["R3 maxima"].std()]
# finds the statistical distributions (mean and std)

x = 2.375/16.0625 # L
b = 1-x # L
L = lengthShaft
# explicitly calculates the length ratio

kDamp = (6*youngMod*inertia*L) / (b*L*x*L * (np.power(L, 2) - np.power(b*L, 2) - np.power(x*L, 2)))
# calculates the spring constant

delta = np.log(maxArray.at[0, "H3 maxima"] / maxArray.at[1, "H3 maxima"])
# calculates logarithmic decrement

zetaExp = delta / np.sqrt(4*np.power(np.pi, 2) - np.power(delta, 2))
# calculates damping ratio

natF = np.sqrt(kDamp/massWheel)
dampF = natF * np.sqrt(1-np.power(zetaExp, 2))
# finds theoretical natural and damped frequency

dampFexp = 2*np.pi / H3Stats[0]
natFexp = dampFexp / np.sqrt(1-np.power(zetaExp, 2))
# finds experimental natural and damped frequency

resF = natF * np.sqrt(1 - 2*np.power(zetaExp, 2))
resFexp = 2*np.pi / R3Stats[0]
# finds experimental/theoretical resonant frequency

lamda = dampFexp / natFexp
BANames = list(["BR11", "BR12", "BR13", "BR14", "BR15", "AR11", "AR12", "AR13", "AR14", "AR15", "R3"])
for i in range(0, len(BANames)):
    ecc = (kDamp * dispInmmCut[BANames[i]] * np.max(dispInmmCut[BANames[i]]) * \
        np.sqrt(np.power(1-np.power(lamda, 2), 2) + np.power(2 * zetaExp * lamda, 2))) \
        / (massWheel * np.power(dampFexp, 2))
    ecc = ecc.dropna()
    eccArray3 = pandas.concat([eccArray3, ecc], axis=1)
# calculates eccentricity for each dataset in this trial


print("H3")
print("k = " + str(kDamp))
print("\u03B4 = " + str(delta))
print("\u03B6 = " + str(zetaExp))
print("theo. \u03C9_n = " + str(natF))
print("theo. \u03C9_d = " + str(dampF))
print("exp. \u03C9_n = " + str(natFexp))
print("exp. \u03C9_d = " + str(dampFexp))
print("")
print("R3")
print("theo. \u03C9_r = " + str(resF))
print("exp. \u03C9_r = " + str(resFexp))
print("")


print("Statistics:")
Stats = pandas.DataFrame([H1Stats, H2Stats, H3Stats, R1Stats, R2Stats, R3Stats], \
    index=["H1", "H2", "H3", "R1", "R2", "R3"], \
    columns=["dt (mean)", "dt (std)", "dy (mean)", "dy (std)"])
print(Stats)
print("")


print("Trial 1 ecc = " + str(eccArray1.mean().mean()))
print("Trial 2 ecc = " + str(eccArray2.mean().mean()))
print("Trial 3 ecc = " + str(eccArray3.mean().mean()))
print("")
print("More data in the generated CSVs!")


eccArray = pandas.concat([eccArray1, eccArray2, eccArray3], axis=1)
eccArray = eccArray.reset_index(drop=True)
# combine the ecc arrays from each trial and make them presentable

eccArray.to_csv("raw_ecc.csv")
maxArray.to_csv("local_maxima.csv")
maxDiffArray.to_csv("local_maxima_diff.csv")
# export all the data arrays to CSV


plt.show()
# open all the previously plotted figures
