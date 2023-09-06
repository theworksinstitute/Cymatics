# Cymatics
In this cymatics project library resonance patterns are analyzed versius various parameters. 
## Contents
- <a href="https://github.com/csanadm/cymatics/blob/main/analyze.py">analyze.py</a>: analyzing an Excel file with `DataFrame` (from `pandas`), by investigating dependence of the symmetry-fold on the parameters, and doing a linear regression to see r-values.
- <a href="https://github.com/csanadm/cymatics/blob/main/fit.py">fit.py</a>: perform an actual fit with `lmfit`, to see if there is a linear form based on the parameters that successfully predicts the symmetry-fold.
- <a href="https://github.com/csanadm/cymatics/blob/main/fit_frequency.py">fit_frequency.py</a>: sama as <a href="https://github.com/csanadm/cymatics/blob/main/fit.py">fit.py</a>, but for frequency prediction.
- <a href="https://github.com/csanadm/cymatics/blob/main/linreg_scikit.py">linreg_scikit.py</a>: perform ML-based linear regression with `scikit-learn`.
- <a href="https://github.com/csanadm/cymatics/blob/main/svr_scikit.py">svr_scikit.py</a>: perform Support Vector Regression with `scikit-learn`, to predict symmetry-fold.
- <a href="https://github.com/csanadm/cymatics/blob/main/svr_scikit_frequency.py">svr_scikit_frequency.py</a>: same as <a href="https://github.com/csanadm/cymatics/blob/main/svr_scikit.py">svr_scikit.py</a>, but for predicting frequency.
- <a href="https://github.com/csanadm/cymatics/blob/main/frequency_plot.py">frequency_plot.py</a>: Plot symmetry-fold vs frequency in various groupings.
- <a href="https://github.com/csanadm/cymatics/blob/main/amp_plots.py">amp_plots.py</a>: Plot amplitudes vs frequency in various groupings.
- <a href="https://github.com/csanadm/cymatics/blob/main/resistance_plots.py">resistance_plots.py</a>: Plot symmetryfold vs resistance in various groupings.
- <a href="https://github.com/csanadm/cymatics/blob/main/resistance_timedep.py">resistance_timedep.py</a>: Analyze the change of resistance with time.
- <a href="https://github.com/csanadm/cymatics/blob/main/input_dependence.py">input_dependence.py</a>: Analyze dependence on input type (analogue/computer) or method (automatic/manual).
- <a href="https://github.com/csanadm/cymatics/blob/main/input_dependency_symm_vs_freq_and_amp.py">input_dependency_symm_vs_freq_and_amp.py</a>: Same as <a href="https://github.com/csanadm/cymatics/blob/main/input_dependence.py">input_dependence.py</a> but for symmetry fold versus frequency and amplitude.
- <a href="https://github.com/csanadm/cymatics/blob/main/pca.py">pca.py</a>: Perform a principal components analysis.

Here is an example plot for the linear fit (two methods: lmfit and scikit/sklearn):

<img alt="fit_residuals" src="https://user-images.githubusercontent.com/38218165/232236561-4d180456-c91a-4e3a-855f-806b7bbf5dcd.png" width="350" /> <img alt="ML_fit_residuals" src="https://user-images.githubusercontent.com/38218165/232236617-15a0535a-f31b-47df-8f22-746e21f05230.png" width="350" />

Here is an example output for the SVR code:

<img alt="scikit_RBF" src="https://user-images.githubusercontent.com/38218165/232236734-217e7816-9af9-46a9-b806-e4feb8756a96.png" width="350" />

Here is an amplitude vs frequency plot, showing boxes from amplitude minimum to amplitude maximum, at each frequency:

<img alt="amplitude_vs_frequency_singlerange" src="https://user-images.githubusercontent.com/38218165/239747308-0ff7bb74-8240-4347-8c17-1af3752e884a.png" width="350" />

And the amplitudes vs frequency, grouped by symmetryfold:

<img alt="V1min_vs_frequency_by_symm" src="https://user-images.githubusercontent.com/38218165/232236981-8fc7a6e6-aeaf-46fe-b244-7036b53c1fcf.png" width="350" /><img alt="V1Max_vs_frequency_by_symm" src="https://user-images.githubusercontent.com/38218165/232236974-7d01fa1c-0bf3-40e2-996c-9867a5a7b63c.png" width="350" />

Finally, symmetry fold versus resistance plots, before and after the cymascope measurement, for the EZ and SZ zones:

<img alt="RBefSZEZ.png" src="https://github.com/csanadm/cymatics/assets/38218165/bfeb0ee0-35c9-4cad-b89b-f4bb1652f3ef" width="350" /><img alt="RSZAftBef.png" src="https://github.com/csanadm/cymatics/assets/38218165/60f98860-ec59-48ec-9e00-a57bc7e3cd6b" width="350" />
