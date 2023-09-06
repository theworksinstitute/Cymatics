import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

markers = ["o", "*", "x", "+", "s", "d", "|", "1", "2"] #, "_"] #NOT 10 MARKES AS THERE ARE ALSO 10 COLORS

def linear_regression_calc(xvector, yvector):
  npoints = np.size(xvector)
  mean_x = np.mean(xvector)
  mean_y = np.mean(yvector)
  SS_yy = np.sum(yvector*yvector) - npoints*mean_y*mean_y
  SS_xy = np.sum(yvector*xvector) - npoints*mean_x*mean_y
  SS_xx = np.sum(xvector*xvector) - npoints*mean_x*mean_x
  if(SS_xx==0 or SS_yy==0 or npoints<=2): return (mean_y, 0, -0.999) #No meaningful r-value, don't care about b0 and b1 either
  b1 = SS_xy/SS_xx
  b0 = mean_y - b1*mean_x
  resvector = yvector - b0 - b1*xvector
  SS_res = np.sum(resvector*resvector)
  rvalue = 1 - SS_res/SS_yy
  return (b0, b1, rvalue)

#df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")

plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Symmetry-fold")
plt.title("Symmetry-fold vs frequency")
xvector = df["Frequency [Hz]"]
yvector = df["Symm1"]
xy_pairs = list(zip(xvector, yvector))
xy_pair_counts = {pair: xy_pairs.count(pair) for pair in xy_pairs}
marker_sizes = [20*xy_pair_counts[pair] for pair in xy_pairs]
plt.scatter(xvector, yvector, s=marker_sizes)
plt.savefig("symmetryfold_vs_frequency.png")

#columns = df.columns
#print(columns)
mycolumns = ["Duration [min]", "Humidity [%]", "Air pressure [mb]", "Water temp. [⁰C]", "Air temp. [⁰C]", "Moon illumination", "V1Min", "V1Max", "V2Min", "V2Max", "V3Min", "V3Max"]
Ncolumns = len(mycolumns)

Nmaxvals = 10
for field in mycolumns:
  uniquevalues = df[field].unique()
  Nvals = len(uniquevalues)
  if(Nvals>Nmaxvals): Nmaxvals = Nvals
print("Maximum number of values: " + str(Nmaxvals))
rvalues = np.zeros([Ncolumns,Nmaxvals])

ifield = 0
for field in mycolumns:
  plt.figure()
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("Symmetry-fold")
  print("Working on " + field + "...")
  uniquevalues = sorted(df[field].unique())
  ival = 0
  Nvals = len(uniquevalues)
  print(str(Nvals) + " unique values for " + field)
  if(Nvals>Nmaxvals): Nmaxvals = Nvals
  for val in uniquevalues:
    df_filtered = df[df[field]==val]
    xvector = df_filtered["Frequency [Hz]"]
    yvector = df_filtered["Symm1"]
    plt.plot(xvector, yvector, marker=markers[ival % len(markers)], linestyle='None', label=field+' = '+str(val))
    linregrpars = linear_regression_calc(xvector,yvector)
    rvalues[ifield][ival] = linregrpars[2]
    ival += 1
  plt.legend()
  shortfield = field.split(' ')[0]
  titlefield = shortfield
  if(len(field.split(' '))>2):
    titlefield += " " + field.split(' ')[1]
    shortfield += "_" + field.split(' ')[1]
    shortfield = shortfield.replace('.','')
  plt.title("Symmetry-fold vs frequency for various " + titlefield + " values")
  ax = plt.subplot(111)
  box = ax.get_position()
  ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.70,box.height*1.06])
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig(shortfield + "_vs_frequency.png")
  ifield += 1

plt.figure()
plt.title("Regression r-value for various parameters")
plt.xlabel("Parameter value [#]")
plt.ylim(0,1.05)
for ifield in range(Ncolumns):
  plt.plot(range(0,Nmaxvals), rvalues[ifield, :], marker=markers[ifield % len(markers)], linestyle='None', label=mycolumns[ifield])
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*1.10,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(0.65, 0.8))
plt.savefig("rvalues_vs_ivalue.png")

#plt.show()