import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters, Parameter, fit_report, report_fit

def mychi2fun(params, df, columns):
  values = np.array(list(params.valuesdict().values()))
  #residuals = df.apply(lambda row: (np.dot(values, np.nan_to_num(row[columns],nan=0)) - np.nan_to_num(row['Symm1'],nan=0))**2, axis=1)
  residuals = []
  for index, row in df.iterrows():
    calc = np.dot(values, row[columns].values)
    meas = row['Frequency [Hz]']
    resid = (calc-meas)**2
    if(np.isnan(resid)): resid=0
    residuals.append(resid)
  residuals = pd.Series(residuals)
  return residuals
  

df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
#df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")
df = df[~df['Symm1'].isna()]
df['V1Min'].fillna(0, inplace=True) # Replace empty V1min values with 0
df['V1Max'].fillna(df['V1Min'], inplace=True) # Replace empty V1Max values with V1Min

mycolumns = ["Duration [min]", "Humidity [%]", "Air pressure [mb]", "Water temp. [⁰C]", "Air temp. [⁰C]", "Moon illumination", "V1Min", "V1Max"]
Ncolumns = len(mycolumns)

params = Parameters()
for field in mycolumns:
  paramname = field.split(' ')[0]
  if(len(field.split(' '))>2):
    paramname += "_" + field.split(' ')[1]
  paramname = paramname.replace('.','')
  paramname += "_multiplier"
  params.add(paramname, value = 0.0)
params.pretty_print()

result = minimize(mychi2fun, params, args=(df,mycolumns,), method='leastsq')
result.params.pretty_print()

print("fcncalls = %i"      % (result.nfev))
print("chi2 = %.3e"        % (result.chisqr))
print("NDF = %i - %i = %i" % (result.ndata,result.nvarys,result.ndata-result.nvarys))
for name in params.keys():
  print("%s = %f +- %f"       % (name, float(result.params[name].value), float(result.params[name].stderr)))

calcs = []
meass = []
for index, row in df.iterrows():
  values = np.array(list(result.params.valuesdict().values()))
  calc = np.dot(values, row[mycolumns].values)
  meas = row['Frequency [Hz]']
  resid = calc-meas
  chi2 = resid**2
  calcs.append(calc)
  meass.append(meas)
  print("%i:   %.2f  ---   %i    (resid = %.2f, chi2 = %.2f)" % (index, calc, meas, resid, chi2))
  
plt.figure()
plt.title("Linear regression fit quality check")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Measurement #")
plt.plot(meass, marker='P', linestyle='none', label='Measured value')
plt.plot(calcs, marker='o', linestyle='none', label='Regression calculation')
plt.legend(loc='upper left')
ax = plt.subplot(111)
#ax.set_ylim(-0.7,18)
#plt.yticks(np.arange(0,18,2))
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*1.15,box.height*1.05])
plt.savefig("fit_frequency_residuals.png")
  