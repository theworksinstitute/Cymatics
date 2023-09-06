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
  #if(rvalue>0.5):
  #  print(xvector)
  #  print(yvector)
  return (b0, b1, rvalue)

#df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")
df = df[~df['V1Min'].isna()] # Remove lines with empty V1Min
df['V1Max'].fillna(df['V1Min'], inplace=True) # Replace empty V1Max values with V1Min

frequencies = sorted(df["Frequency [Hz]"].unique())
Nfreqs = len(frequencies)

print("##################### INPUT TYPE ####################")

for input in ["computer", "analogue"]:
  df_firstfilter = df[df["Audio input"]==input]
  xvector = df_firstfilter["V1Min"].to_numpy()
  yvector = df_firstfilter["Symm1"].to_numpy()
  if(len(xvector[xvector>0])>2): print("Overall r-value, " + input + " input, V1Min vs symm: " + str(linear_regression_calc(xvector,yvector)[2]))
  xvector = df_firstfilter["V1Max"].to_numpy()
  yvector = df_firstfilter["Symm1"].to_numpy()
  if(len(xvector[xvector>0])>2): print("Overall r-value, " + input + " input, V1Max vs symm: " + str(linear_regression_calc(xvector,yvector)[2]))
  for field in ["V1Min", "V1Max"]:
    print("############### " + input + " " + field + " ###############")
    plt.figure()
    ifreq = 0
    for f in frequencies:
      df_filtered = df_firstfilter[df_firstfilter["Frequency [Hz]"]==f]
      xvector = df_filtered[field].to_numpy()
      yvector = df_filtered["Symm1"].to_numpy()
      plt.plot(xvector, yvector, marker=markers[ifreq % len(markers)], linestyle='None', label='f='+str(f)+' Hz')
      if(len(xvector[xvector>0])>2): print(str(f) + ' Hz -> ' + str(linear_regression_calc(xvector,yvector)[2]))
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
    plt.title(input + " input")
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.92,box.height*1.06])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig(shortfield + "_" + input + ".png")
		
print("##################### AMPLITUDE CHANGE ####################")

for ampchange in ["manual", "automated"]:
  df_firstfilter = df[df["Amplitude change"]==ampchange]
  xvector = df_firstfilter["V1Min"].to_numpy()
  yvector = df_firstfilter["Symm1"].to_numpy()
  if(len(xvector[xvector>0])>2): print("Overall r-value, " + ampchange + " amp. change, V1Min vs symm: " + str(linear_regression_calc(xvector,yvector)[2]))
  xvector = df_firstfilter["V1Max"].to_numpy()
  yvector = df_firstfilter["Symm1"].to_numpy()
  if(len(xvector[xvector>0])>2): print("Overall r-value, " + ampchange + " amp. change, V1Max vs symm: " + str(linear_regression_calc(xvector,yvector)[2]))
  for field in ["V1Min", "V1Max"]:
    print("############### " + ampchange + " " + field + " ###############")
    plt.figure()
    ifreq = 0
    for f in frequencies:
      df_filtered = df_firstfilter[df_firstfilter["Frequency [Hz]"]==f]
      xvector = df_filtered[field].to_numpy()
      yvector = df_filtered["Symm1"].to_numpy()
      plt.plot(xvector, yvector, marker=markers[ifreq % len(markers)], linestyle='None', label='f='+str(f)+' Hz')
      if(len(xvector[xvector>0])>2): print(str(f) + ' Hz -> ' + str(linear_regression_calc(xvector,yvector)[2]))
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
    plt.title(ampchange + " amplitude change")
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05,box.y0,box.width*0.92,box.height*1.06])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig(shortfield + "_" + ampchange + ".png")

