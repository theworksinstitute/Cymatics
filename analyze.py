import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

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
  #if(rvalue>0.5):
  #  print(xvector)
  #  print(yvector)
  return (b0, b1, rvalue)

#df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")

NUMBER_OF_MIN_POINTS_FOR_REGRESSION = 2

filter = ""
if(len(sys.argv)>1): filter = sys.argv[1]
# Filter data before everything on input type or amplitude change
if(filter=="computer" or filter=="analogue"):
  input = filter
  df = df[df["Audio input"]==input]
if(filter=="manual" or filter=="automated"):
  ampchange = filter
  df = df[df["Amplitude change"]==ampchange]

columns = df.columns
print(columns)
mycolumns = ["Duration [min]", "Humidity [%]", "Air pressure [mb]", "Water temp. [⁰C]", "Air temp. [⁰C]", "Moon illumination", "V1Min", "V1Max", "V2Min", "V2Max", "V3Min", "V3Max"]
Ncolumns = len(mycolumns)

############################################################
############### CORR WITH SYMM FOR EACH FREQ ###############
############################################################

frequencies = sorted(df["Frequency [Hz]"].unique())
Nfreqs = len(frequencies)

rvalues = np.zeros([Ncolumns,Nfreqs])

ifield = 0
for field in mycolumns:
  plt.figure()
  ifreq = 0
  print("Working on " + field + " versus frequency...")
  for f in frequencies:
    df_filtered = df[df["Frequency [Hz]"]==f]
    xvector = df_filtered[field].to_numpy()
    yvector = df_filtered["Symm1"].to_numpy()
    plt.plot(xvector, yvector, marker=markers[ifreq % len(markers)], linestyle='None', label='f='+str(f)+' Hz')
    if(len(xvector[xvector>0])>NUMBER_OF_MIN_POINTS_FOR_REGRESSION):
      linregrpars = linear_regression_calc(xvector,yvector)
      #print("f = " + str(f) + " Hz -> " + "{:.3f}".format(linregrpars[2]))
      rvalues[ifield][ifreq] = linregrpars[2]
      if(np.isnan(rvalues[ifield][ifreq])): rvalues[ifield][ifreq] = 0
    ifreq += 1
  plt.legend()
  plt.xlabel(field)
  shortfield = field.split(' ')[0]
  titlefield = shortfield
  if(len(field.split(' '))>2):
    titlefield += " " + field.split(' ')[1]
    shortfield += "_" + field.split(' ')[1]
    shortfield = shortfield.replace('.','')
  plt.ylabel("Symmetry-fold")
  plt.title("Symmetry-fold vs " + titlefield + " for each frequency")
  ax = plt.subplot(111)
  box = ax.get_position()
  ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.92,box.height*1.06])
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  #plt.savefig(shortfield + filter + ".png")
  ifield += 1

plt.figure()
plt.title("Regression r-value versus frequency")
plt.xlabel("Frequency [Hz]")
plt.ylim(0,1.05)
for ifield in range(Ncolumns):
  plt.plot(frequencies, rvalues[ifield, :], marker=markers[ifield % len(markers)], linestyle='None', label=mycolumns[ifield])
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.80,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#plt.savefig("rvalues" + filter + ".png")

plt.figure()
plt.title("Average regression r-value for symm-fold versus variable")
plt.xlabel("")
#plt.ylim(-0.01,0.2)
averages = np.average(rvalues, axis=1)
print(len(mycolumns))
print(mycolumns)
print(len(averages))
print(averages)
plt.plot(mycolumns, averages, marker=markers[0], linestyle='None')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.xticks(rotation=45, ha='right')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.04,box.y0+box.height*0.17,box.width*1.15,box.height*0.90])
#plt.savefig("rvalues_averages" + filter + ".png")

############################################################
############ CORR WITH FREQ FOR EACH SYMM1 #################
############################################################

symmvals = sorted(df["Symm1"].unique())
Nsymmvals = len(symmvals)

rvalues2 = np.zeros([Ncolumns,Nsymmvals])

ifield = 0
for field in mycolumns:
  plt.figure()
  isymm = 0
  print("Working on " + field + " versus symmetry-fold...")
  for symm in symmvals:
    df_filtered = df[df["Symm1"]==symm]
    xvector = df_filtered[field].to_numpy()
    yvector = df_filtered["Frequency [Hz]"].to_numpy()
    plt.plot(xvector, yvector, marker=markers[isymm % len(markers)], linestyle='None', label='symm='+str(symm))
    if(len(xvector[xvector>0])>NUMBER_OF_MIN_POINTS_FOR_REGRESSION):
      linregrpars = linear_regression_calc(xvector,yvector)
      #print("f = " + str(f) + " Hz -> " + "{:.3f}".format(linregrpars[2]))
      rvalues2[ifield][isymm] = linregrpars[2]
      if(np.isnan(rvalues2[ifield][isymm])): rvalues2[ifield][isymm] = 0
    isymm += 1
  plt.legend()
  plt.xlabel(field)
  shortfield = field.split(' ')[0]
  titlefield = shortfield
  if(len(field.split(' '))>2):
    titlefield += " " + field.split(' ')[1]
    shortfield += "_" + field.split(' ')[1]
    shortfield = shortfield.replace('.','')
  plt.ylabel("Frequency [Hz]")
  plt.title("Frequency vs " + titlefield + " and symmetry-fold")
  ax = plt.subplot(111)
  box = ax.get_position()
  ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.92,box.height*1.06])
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  #plt.savefig(shortfield + filter + "_symmplot.png")
  ifield += 1

plt.figure()
plt.title("Regression r-value versus symmetry-fold")
plt.xlabel("Symmetry-fold")
plt.ylim(0,1.05)
for ifield in range(Ncolumns):
  plt.plot(symmvals, rvalues2[ifield, :], marker=markers[ifield % len(markers)], linestyle='None', label=mycolumns[ifield])
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.80,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#plt.savefig("rvalues_symm" + filter + ".png")

plt.figure()
plt.title("Average regression r-value for frequency versus variable")
plt.xlabel("")
#plt.ylim(-0.1,0.1)
averages2 = np.average(rvalues2, axis=1)
print(len(mycolumns))
print(mycolumns)
print(len(averages2))
print(averages2)
plt.plot(mycolumns, averages2, marker=markers[0], linestyle='None')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.xticks(rotation=45, ha='right')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.04,box.y0+box.height*0.17,box.width*1.15,box.height*0.90])
#plt.savefig("rvalues_symm_averages" + filter + ".png")



############################################################
############ CORR WITH V1MIN FOR EACH FREQ #################
############################################################

rvalues = np.zeros([Ncolumns,Nfreqs])

ifield = 0
for field in mycolumns:
  plt.figure()
  ifreq = 0
  print("Working on" + field + " versus V1Min...")
  for f in frequencies:
    df_filtered = df[df["Frequency [Hz]"]==f]
    xvector = df_filtered[field].to_numpy()
    yvector = df_filtered["V1Min"].to_numpy()
    plt.plot(xvector, yvector, marker=markers[ifreq % len(markers)], linestyle='None', label='f='+str(f)+' Hz')
    if(len(xvector[xvector>0])>NUMBER_OF_MIN_POINTS_FOR_REGRESSION):
      linregrpars = linear_regression_calc(xvector,yvector)
      #print("f = " + str(f) + " Hz -> " + "{:.3f}".format(linregrpars[2]))
      rvalues[ifield][ifreq] = linregrpars[2]
      if(np.isnan(rvalues[ifield][ifreq])): rvalues[ifield][ifreq] = 0
    ifreq += 1
  plt.legend()
  plt.xlabel(field)
  shortfield = field.split(' ')[0]
  titlefield = shortfield
  if(len(field.split(' '))>2):
    titlefield += " " + field.split(' ')[1]
    shortfield += "_" + field.split(' ')[1]
    shortfield = shortfield.replace('.','')
  plt.ylabel("amplitude")
  plt.title("amplitude vs " + titlefield + " for each frequency")
  ax = plt.subplot(111)
  box = ax.get_position()
  ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.92,box.height*1.06])
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig(shortfield + filter + "_V1minplot.png")
  ifield += 1

plt.figure()
plt.title("Regression r-value versus amplitude")
plt.xlabel("Frequency [Hz]")
plt.ylim(0,1.05)
for ifield in range(Ncolumns):
  plt.plot(frequencies, rvalues[ifield, :], marker=markers[ifield % len(markers)], linestyle='None', label=mycolumns[ifield])
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.80,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig("rvalues" + filter + "_V1minplot.png")

plt.figure()
plt.title("Average regression r-value for symm-fold versus variable")
plt.xlabel("")
#plt.ylim(-0.01,0.2)
averages = np.average(rvalues, axis=1)
print(len(mycolumns))
print(mycolumns)
print(len(averages))
print(averages)
plt.plot(mycolumns, averages, marker=markers[0], linestyle='None')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
plt.xticks(rotation=45, ha='right')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.04,box.y0+box.height*0.17,box.width*1.15,box.height*0.90])
plt.savefig("rvalues_averages" + filter + "V1minplot.png")