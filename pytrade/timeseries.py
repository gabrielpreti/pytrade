from pytrade.backtesting.backtest import GoogleFinanceBacktest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import statsmodels.tsa.stattools as stattools
from statsmodels.graphics import tsaplots
import statsmodels.api as smapi
import statsmodels.stats.diagnostic as diagnostic
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model

import statsmodels.tsa.api as smt
import scipy.stats as scs

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        smapi.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
feed = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed.loadAll()

feed2016 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2016, debugMode=False,
                                     csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed2016.loadAll()

closevalues = pd.Series(np.array(feed.getDataSeries(instrument=codes[0]).getCloseDataSeries()))
tsaplots.plot_acf(closevalues.diff().dropna(), alpha=0.05, lags=40)

#################################################################################################
#Correlograms for the log returns
#################################################################################################
with PdfPages('/tmp/correlograms.pdf') as pdf:
    lines = 2
    columns = 1
    fig = plt.figure()
    ar=1
    for code in codes:
        if ar>lines*columns:
            pdf.savefig(fig)
            fig = plt.figure()
            ar=1
        fig.add_subplot(lines, columns, ar)
        closevalues = pd.Series(np.array(feed.getDataSeries(instrument=code).getCloseDataSeries()))
        tsaplots.plot_acf(np.log(closevalues).diff().dropna(), alpha=0.05, lags=40, ax=fig.gca(), title=code)
        ar += 1
    pdf.savefig(fig)
    plt.close('all')

#################################################################################################
#Trying to fit an ARMA model
#################################################################################################
stattools.q_stat()
stattools.acf()

closevalues = pd.Series(np.array(feed.getDataSeries(instrument=codes[1]).getCloseDataSeries()))
# arma = smapi.tsa.ARMA(np.array(np.log(closevalues).diff().dropna()), (0, 5)).fit(maxiter=10000)
arma = smapi.tsa.ARMA(np.array(closevalues), (0, 5)).fit(maxiter=10000)
pvalue = diagnostic.acorr_ljungbox(arma.resid, lags=[20])[1]
print(pvalue)

results = {}
for code in codes:
    final_arma = None
    final_aic = 1e200
    final_order = None
    for ar in range(1, 5):
        for ma in range(1, 5):
            try:
                closevalues = pd.Series(np.array(feed.getDataSeries(instrument=code).getCloseDataSeries()))
                curr_arma = smapi.tsa.ARMA(np.array(np.log(closevalues).diff().dropna()), (ar, ma)).fit(maxiter=10000) #fit an ARMA model
                if curr_arma.aic < final_aic: #Find the model with whe lowest AIC
                    final_aic = curr_arma.aic
                    final_arma = curr_arma
                    final_order = (ar, ma)
            except Exception:
                print("Exception processing code %s with values (%d, %d)" % (code, ar, ma))
                continue
    pvalue = diagnostic.acorr_ljungbox(final_arma.resid, lags=[20])[1] #Execute the Ljung-Box test in the residuals of the best model found (in terms of AIC)
    if pvalue > 0.05: #Verify if the residuals is a white noise process with 95% confidence. If it is, we may have a good fit
        print("Residuals for code %s with values (%d, %d) can be white noise with p-value %s" % (code, final_order[0], final_order[1], pvalue))
        results[code] = final_arma
    else:
        print("Discarding %s" % (code))

#################################################################################################
#Trying to fit an ARIMA model
#################################################################################################
results = {}
for code in codes:
    final_arima = None
    final_order = None
    for ar in range(1, 10):
        for integrated in range(1, 10):
            for ma in range(1, 10):
                try:
                    # closevalues = pd.Series(np.array(feed.getDataSeries(instrument=code).getCloseDataSeries()), index=np.array(feed.getDataSeries(instrument=code).getDateTimes()))
                    # closevalues = closevalues.resample('D').last()
                    closevalues = np.array(feed.getDataSeries(instrument=code).getCloseDataSeries())
                    cur_arima = ARIMA(np.array(closevalues), (ar, integrated, ma)).fit(disp=0, maxiter=10000)
                    if final_arima is None or cur_arima.aic<final_arima.aic:
                        final_arima = cur_arima
                        final_order = (ar, integrated, ma)
                except: continue
    pvalue = diagnostic.acorr_ljungbox(final_arima.resid, lags=[40])[1] if final_arima is not None else 0 # Execute the Ljung-Box test in the residuals of the best model found (in terms of AIC)
    if pvalue > 0.05: #Verify if the residuals is a white noise process with 95% confidence. If it is, we may have a good fit
        print("Residuals for code %s with values (%d, %d, %d) can be white noise with p-value %s" % (code, final_order[0], final_order[1], final_order[2], pvalue))
        results[code] = final_arima
    else:
        print("Discarding %s" % (code))

n_steps=30
# sample_dates = pd.Series(np.array(feed.getDataSeries(instrument=codes[0]).getDateTimes()), index=np.array(feed.getDataSeries(instrument=codes[0]).getDateTimes())).resample('W').last()
variations = pd.DataFrame(columns=('code', 'forecast', 'real'))
for code in results.keys():
    model = results[code]
    f, err95, ci95 = model.forecast(steps=n_steps)  # 95% CI
    _, err99, ci99 = model.forecast(steps=n_steps, alpha=0.01)  # 99% CI
    # closevalues2016 = pd.Series(np.array(feed2016.getDataSeries(instrument=code).getCloseDataSeries()),
    #                             index=np.array(feed2016.getDataSeries(instrument=code).getDateTimes())).resample(
    #     'W').last()
    closevalues2016 = np.array(feed2016.getDataSeries(instrument=code).getCloseDataSeries())

    # idx = pd.DatetimeIndex(np.array(pd.Series(feed2016.getDataSeries(instrument=code).getDateTimes(),
    #                                           index=feed2016.getDataSeries(instrument=code).getDateTimes()).resample(
    #     'W').last())[:n_steps])
    idx = pd.DatetimeIndex(np.array(feed2016.getDataSeries(instrument=code).getDateTimes())[:n_steps])
    fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
    fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
    fc_all = fc_95.combine_first(fc_99)
    fc_all['real_values'] = closevalues2016[:n_steps]

    # series2015 = pd.Series(np.array(feed.getDataSeries(instrument=code).getCloseDataSeries()),
    #                             index=np.array(feed.getDataSeries(instrument=code).getDateTimes())).resample(
    #     'W').last()
    series2015 = pd.Series(np.array(feed.getDataSeries(instrument=code).getCloseDataSeries()),
                           index=np.array(feed.getDataSeries(instrument=code).getDateTimes()))
    fc_all = fc_all.append(pd.DataFrame({'forecast':0, 'lower_ci_95':0, 'lower_ci_99': 0, 'upper_ci_95': 0, 'upper_ci_99': 0, 'real_values': series2015[-1]}, index=series2015.index[-1:]))
    fc_all.sort(inplace=True)
    variations = variations.append({'code': code, 'real': (fc_all.real_values[-1]-fc_all.real_values[0])/fc_all.real_values[0], 'forecast': (fc_all.forecast[-1]-fc_all.real_values[0])/fc_all.real_values[0]}, ignore_index=True)


code = codes[9]
model = results[code]
# tsplot(model.resid, lags=30)

f, err95, ci95 = model.forecast(steps=n_steps) # 95% CI
_, err99, ci99 = model.forecast(steps=n_steps, alpha=0.01) # 99% CI
closevalues2016 = pd.Series(np.array(feed2016.getDataSeries(instrument=code).getCloseDataSeries()), index=np.array(feed2016.getDataSeries(instrument=code).getDateTimes())).resample('W').last()

# idx = pd.date_range(sample_dates[-1], periods=n_steps, freq='D')
idx = pd.DatetimeIndex(np.array(pd.Series(feed2016.getDataSeries(instrument=code).getDateTimes(), index=feed2016.getDataSeries(instrument=code).getDateTimes()).resample('W').last())[:n_steps])
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all['real_values'] = closevalues2016[:n_steps].values

plt.style.use('bmh')
fig = plt.figure(figsize=(9,7))
ax = plt.gca()

styles = ['b-', '0.2', '0.75', '0.2', '0.75', '-']
fc_all.plot(ax=plt.gca(), style=styles, label='Real')
# plt.plot(idx, real_values, label='Real values')
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)

plt.title(code)
plt.legend(loc='best', fontsize=10)

#Start section 10.3
#################################################################################################
#Trying to fit an GARCH model
#################################################################################################

