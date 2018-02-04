import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

intermediary_results_file = "/export/pytrade/intermediary_results.csv"

results = pd.read_csv(intermediary_results_file)
results.dropna(inplace=True)
results['profit']=(results.final_equity-results.initial_equity)/results.initial_equity


sns.distplot(results.final_equity)
sns.jointplot(x="training_buy_window_days", y="final_equity", data=results, kind='reg', bins=6)
sns.jointplot(x="forecast_buy_window_days", y="final_equity", data=results, kind='reg')
sns.jointplot(x="training_sell_window_days", y="final_equity", data=results, kind='reg')
sns.jointplot(x="forecast_sell_window_days", y="final_equity", data=results, kind='reg')


sns.jointplot(x="training_buy_window_days", y="final_equity", data=results, kind="hex")
sns.jointplot(x="forecast_buy_window_days", y="final_equity", data=results, kind="hex")
sns.jointplot(x="training_sell_window_days", y="final_equity", data=results, kind="hex")
sns.jointplot(x="forecast_sell_window_days", y="final_equity", data=results, kind="hex")

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1)
sns.kdeplot(results.training_buy_window_days, results.final_equity, cmap=cmap, n_levels=60, shade=True);




plt.close('all')
fig = plt.figure()

fig.add_subplot(2, 2, 1)
for d in list(results.training_buy_window_days.unique()):
    sns.kdeplot(results.loc[results['training_buy_window_days']==d].final_equity, label="training_buy_window_days=%s"%(d), shade=False, kernel='gau')
plt.legend()

fig.add_subplot(2, 2, 2)
for d in list(results.forecast_buy_window_days.unique()):
    sns.kdeplot(results.loc[results['forecast_buy_window_days']==d].final_equity, label="forecast_buy_window_days=%s"%(d), shade=False, kernel='gau')
plt.legend()

fig.add_subplot(2, 2, 3)
for d in list(results.training_sell_window_days.unique()):
    sns.kdeplot(results.loc[results['training_sell_window_days']==d].final_equity, label="training_sell_window_days=%s"%(d), shade=False, kernel='gau')
plt.legend()

fig.add_subplot(2, 2, 4)
for d in list(results.forecast_sell_window_days.unique()):
    sns.kdeplot(results.loc[results['forecast_sell_window_days']==d].final_equity, label="forecast_sell_window_days=%s"%(d), shade=False, kernel='gau')
plt.legend()


results.sort(columns='profit', ascending=False)[['training_buy_window_days', 'forecast_buy_window_days', 'training_sell_window_days', 'forecast_sell_window_days', 'final_equity']]
results[['training_buy_window_days', 'forecast_buy_window_days', 'training_sell_window_days', 'forecast_sell_window_days']]



