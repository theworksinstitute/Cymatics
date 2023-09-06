import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import numpy as np

markers = ["o", "*", "^", "v", "s", "d", ">", "<", "h"] #, "_"] #NOT 10 MARKES AS THERE ARE ALSO 10 COLORS

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

times = np.array([0,5,10,20])
def fit_linear_timedep(row,column_code):
  xvector = times
  yvector = row[["R_" + column_code + "_0min [MO]","R_" + column_code + "_5min [MO]","R_" + column_code + "_10min [MO]","R_" + column_code + "_20min [MO]"]]
  b0, b1, rvalue = linear_regression_calc(xvector,yvector)
  return b1/np.mean(yvector)
  #return rvalue


#Res. before SZ [MΩ]	Res. before EZ [MΩ]	Res. after SZ [MΩ]	Res. after EZ [MΩ]

def fit_linear_before_after(row,column_code):
  diff = (row["Res. before " + column_code + " [MΩ]"] - row["Res. after " + column_code + " [MΩ]"])
  mean = (row["Res. before " + column_code + " [MΩ]"] + row["Res. after " + column_code + " [MΩ]"])/2
  rel_diff = diff/mean
  return rel_diff
  #return rvalue
  
#############################################
########## TIMEDEP NO MEASUREMENT ###########
#############################################
  
df = pd.read_excel("EZ_water_measurements.xlsx", sheet_name="Sheet1")

result_SZ = df.apply(fit_linear_timedep,args=('SZ',),axis=1)
result_EZ = df.apply(fit_linear_timedep,args=('EZ',),axis=1)
#print(result)

plt.figure()
plt.xlabel("Measurement #")
plt.ylabel("Resistance slope")
#plt.ylabel("Resistance change, r-value")
plt.ylim([-0.05,0.05])
plt.xticks(np.arange(len(result_SZ)))
plt.plot(result_SZ.index,result_SZ.values,marker='o',linestyle='None',label="SZ water")
plt.plot(result_EZ.index,result_EZ.values,marker='*',linestyle='None',label="EZ water")
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.01,box.y0,box.width*1.07,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.9))
#plt.savefig("resistance_slope.png")
##plt.savefig("resistance_slope_rvalue.png")

mean_SZ = result_SZ.values.mean()
var_SZ = result_SZ.values.var()
mean_EZ = result_EZ.values.mean()
var_EZ = result_EZ.values.var()
print(str(mean_SZ) + " +- " + str(var_SZ))
print(str(mean_EZ) + " +- " + str(var_EZ))
  
#############################################
########### TIMEDEP BEFORE/AFTER ############
#############################################
 
df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")

times = np.array([0,5,10,20])

result_SZ = df.apply(fit_linear_before_after,args=('SZ',),axis=1)
result_EZ = df.apply(fit_linear_before_after,args=('EZ',),axis=1)
#print(result)

plt.figure()
plt.xlabel("Measurement #")
plt.ylabel("Resistance after - before")
#plt.ylim([-0.05,0.05])
plt.xticks(np.arange(len(result_SZ)))
plt.plot(result_SZ.index,result_SZ.values,marker='o',linestyle='None',label="SZ water")
plt.plot(result_EZ.index,result_EZ.values,marker='*',linestyle='None',label="EZ water")
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.01,box.y0,box.width*1.07,box.height*1.06])
ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.9))
plt.savefig("resistance_before_after.png")

mean_SZ = result_SZ.values.mean()
var_SZ = result_SZ.values.var()
mean_EZ = result_EZ.values.mean()
var_EZ = result_EZ.values.var()
print(str(mean_SZ) + " +- " + str(var_SZ))
print(str(mean_EZ) + " +- " + str(var_EZ))