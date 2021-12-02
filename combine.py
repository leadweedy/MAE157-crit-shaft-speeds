# roger truong
# MAE 157


## last tested with Python 3.9.9


import pandas
import glob


## INTERPOLATION OF VOLTAGE -------------------------------------------------------------------
dataFiles = []
for each_file in glob.glob("*.txt"):
    dataFiles.append(each_file)
# reads all the data files ending in *.txt

yDisp = pandas.DataFrame()
# create empty dataframe for storing all 36 y displacements

for i in dataFiles:
    filename = str(i).replace(".txt","").replace("DataG3","")
    #simplify the filename

    df = pandas.read_csv(i, sep="\t", header=None, names=["time", "x", filename]) 
    yDisp = pandas.concat([yDisp, df[filename]], axis=1)
    # add dataset onto dataframe


yDisp.to_csv("data_raw_volts.csv", index=False)
# write all the data to CSV
print("Done!")