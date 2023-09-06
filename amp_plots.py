import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import numpy as np

markers = ["o", "*", "x", "+", "s", "d", "|", "1", "2"] #, "_"] #NOT 10 MARKES AS THERE ARE ALSO 10 COLORS

#df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")
df = df[~df['Symm1'].isna()]

xvector0 = df["Frequency [Hz]"].to_numpy()

yminvec1 = df["V1Min"].to_numpy()
ymaxvec1 = df["V1Max"].to_numpy()
nan_mask1 = np.isnan(ymaxvec1) # OTHER POSSIBILITY: ymaxvec = np.nan_to_num(ymaxvec, nan=-999)
ymaxvec1[nan_mask1] = yminvec1[nan_mask1] # REPLACE MAX WITH MIN IF VMAX=NAN
xvector0 = xvector0[~np.isnan(yminvec1)] # REMOVE POINT WHERE VMIN = NAN
ymaxvec1 = ymaxvec1[~np.isnan(yminvec1)] # REMOVE POINT WHERE VMIN = NAN
yminvec1 = yminvec1[~np.isnan(yminvec1)] # REMOVE POINT WHERE VMIN = NAN

slope, intercept, r_value, p_value, std_err = linregress(xvector0, yminvec1)
print(f"V1Min vs frequency linear regression slope: {slope}, intercept: {intercept}, R-value: {r_value}, P-value: {p_value}, error: {std_err}")

slope, intercept, r_value, p_value, std_err = linregress(xvector0, ymaxvec1)
print(f"V1Max vs frequency linear regression slope: {slope}, intercept: {intercept}, R-value: {r_value}, P-value: {p_value}, error: {std_err}")

yminvec2 = df["V2Min"].to_numpy()
ymaxvec2 = df["V2Max"].to_numpy()
yminvec2 = np.nan_to_num(yminvec2)
nan_mask2 = np.isnan(ymaxvec2)
ymaxvec2[nan_mask2] = yminvec2[nan_mask2] # REPLACE MAX WITH MIN IF VMAX=NAN

yminvec3 = df["V3Min"].to_numpy()
ymaxvec3 = df["V3Max"].to_numpy()
yminvec3 = np.nan_to_num(yminvec3)
nan_mask3 = np.isnan(ymaxvec3)
ymaxvec3[nan_mask3] = yminvec3[nan_mask3] # REPLACE MAX WITH MIN IF VMAX=NAN

plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Amplitude vs frequency")
plt.xlim(40,230)
plt.ylim(0.1,4.5)
for ipoint in range(len(xvector0)):
  #plt.vlines(xvector0[ipoint], yminvec1[ipoint], ymaxvec1[ipoint], colors='r', linewidth=4)
  if(not np.isnan(yminvec1[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec1[ipoint]), 2, ymaxvec1[ipoint]-yminvec1[ipoint], linewidth=1, edgecolor='b', facecolor='none'))
  if(not np.isnan(yminvec2[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec2[ipoint]), 2, ymaxvec2[ipoint]-yminvec2[ipoint], linewidth=1, edgecolor='b', facecolor='none'))
  if(not np.isnan(yminvec3[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec3[ipoint]), 2, ymaxvec3[ipoint]-yminvec3[ipoint], linewidth=1, edgecolor='b', facecolor='none'))
plt.savefig("amplitude_vs_frequency.png")

plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Amplitude vs frequency")
plt.xlim(40,230)
plt.ylim(0.1,4.5)
for ipoint in range(len(xvector0)):
  #plt.vlines(xvector0[ipoint], yminvec1[ipoint], ymaxvec1[ipoint], colors='r', linewidth=4)
  if(not np.isnan(yminvec1[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec1[ipoint]), 2, ymaxvec1[ipoint]-yminvec1[ipoint], linewidth=1, edgecolor='r', facecolor='none'))
  if(not np.isnan(yminvec2[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec2[ipoint]), 2, ymaxvec2[ipoint]-yminvec2[ipoint], linewidth=1, edgecolor='g', facecolor='none'))
  if(not np.isnan(yminvec3[ipoint])): plt.gca().add_patch(Rectangle((xvector0[ipoint]-1,yminvec3[ipoint]), 2, ymaxvec3[ipoint]-yminvec3[ipoint], linewidth=1, edgecolor='b', facecolor='none'))
plt.gca().add_patch(Rectangle((55,3.5), 2, 0.2, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((55,3.2), 2, 0.2, linewidth=1, edgecolor='g', facecolor='none'))
plt.gca().add_patch(Rectangle((55,2.9), 2, 0.2, linewidth=1, edgecolor='b', facecolor='none'))
plt.text(60, 3.55, 'V1 range')
plt.text(60, 3.25, 'V2 range')
plt.text(60, 2.95, 'V3 range')
plt.savefig("amplitude_vs_frequency_threecolors.png")

plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Amplitude vs frequency")
plt.xlim(40,230)
plt.ylim(0.1,4.5)
dfVmin = df[['Frequency [Hz]','V1Min', 'V2Min', 'V3Min']].fillna(value=np.inf)
Vminarray = dfVmin.groupby('Frequency [Hz]').apply(lambda x: x[['V1Min', 'V2Min', 'V3Min']].min().min()).to_numpy()
dfVmax = df[['Frequency [Hz]','V1Max', 'V2Max', 'V3Max']].fillna(value=-np.inf)
Vmaxarray = dfVmax.groupby('Frequency [Hz]').apply(lambda x: x[['V1Max', 'V2Max', 'V3Max']].max().max()).to_numpy()
freqvals = sorted(df['Frequency [Hz]'].unique())
print(Vminarray)
print(Vmaxarray)
print(freqvals)
for ipoint in range(len(Vmaxarray)):
  plt.gca().add_patch(Rectangle((freqvals[ipoint]-1,Vminarray[ipoint]), 2, Vmaxarray[ipoint]-Vminarray[ipoint], linewidth=1, edgecolor='b', facecolor='none'))
plt.savefig("amplitude_vs_frequency_singlerange.png")

Symm1vals = sorted(df["Symm1"].unique())
Nsymmvals = len(Symm1vals)

ival = 0
plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("V1Min")
plt.title("Amplitude vs frequency for each symm-fold")
for Symm1val in Symm1vals:
  print(f"Working on Symm1={Symm1val} versus frequency...")
  df_filtered = df[df["Symm1"]==Symm1val]
  xvector = df_filtered["Frequency [Hz]"]
  yvector = df_filtered["V1Min"]
  plt.plot(xvector, yvector, marker=markers[ival % len(markers)], linestyle='None', label=f"symm={Symm1val}")
  ival += 1
plt.legend()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.03,box.y0,box.width*0.91,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("V1min_vs_frequency_by_symm.png")

ival = 0
plt.figure()
plt.xlabel("Frequency [Hz]")
plt.ylabel("V1Max")
plt.title("Amplitude vs frequency for each symm-fold")
for Symm1val in Symm1vals:
  print(f"Working on Symm1={Symm1val} versus frequency...")
  df_filtered = df[df["Symm1"]==Symm1val]
  xvector = df_filtered["Frequency [Hz]"]
  yvector = df_filtered["V1Max"]
  plt.plot(xvector, yvector, marker=markers[ival % len(markers)], linestyle='None', label=f"symm={Symm1val}")
  ival += 1
plt.legend()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.03,box.y0,box.width*0.91,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("V1Max_vs_frequency_by_symm.png")