from pytrade.base import TradingAlgorithm
from pyalgotrade.broker import Order
import numpy as np
import talib
import pandas as pd



class MLAnalysisTradingAlgorithm(TradingAlgorithm):
    def __init__(self, feed, broker, riskFactor, models, pipeline, training_buy_window_days=60,
                 forecast_buy_window_days=30, training_sell_window_days=60, forecast_sell_window_days=30,
                 buy_result_function=None, sell_result_function=None):
        super(MLAnalysisTradingAlgorithm, self).__init__(feed, broker)
        self.__riskFactor = riskFactor
        self.__models = models #Esse aqui e para o caso de ja passarmos os modelos prontos para cada stock
        self.__dynamic_buy_models = {} # Esse aqui e para os modelos gerados dinamicamente: e uma mapa onde a chave e o stock code, e o elemento e uma tupla (modelo, data)
        self.__dynamic_sell_models = {}  # Esse aqui e para os modelos gerados dinamicamente: e uma mapa onde a chave e o stock code, e o elemento e uma tupla (modelo, data)
        self.__max_dynamic_model_age = 30
        self.__training_buy_window_days = training_buy_window_days
        self.__forecast_buy_window_days = forecast_buy_window_days
        self.__training_sell_window_days = training_sell_window_days
        self.__forecast_sell_window_days = forecast_sell_window_days
        self.__pipeline = pipeline
        self.__purchases = {}
        self.__buy_result_function = buy_result_function
        self.__sell_result_function = sell_result_function
        self.__logger = broker.getLogger()
        broker.getOrderUpdatedEvent().subscribe(self.__onOrderEvent)

    def __onOrderEvent(self, broker_, orderEvent):
        order = orderEvent.getOrder()
        if order.getAction() == Order.Action.BUY and (
                        order.getState() == Order.State.FILLED or order.getState() == Order.State.PARTIALLY_FILLED):
            instrument = order.getInstrument()
            self.__purchases[instrument] = order.getAvgFillPrice()

    def shouldAnalyze(self, bar, instrument):
        if self.getBroker().getShares(instrument) > 0:
            return True

        if self.getBroker().getEquity() < 100:
            return False

        if self.__models is not None and instrument in self.__models.keys():
            return len(self._feed[instrument].getCloseDataSeries()) > 30
        else:
            return len(self._feed[
                           instrument].getCloseDataSeries()) > max(
                (self.__training_buy_window_days + self.__forecast_buy_window_days),
                (self.__training_sell_window_days + self.__forecast_sell_window_days))

    def __get_dynamic_model_date_for_instrument(self, model_repository, instrument):
        return model_repository[instrument][1]

    def __get_dynamic_model_for_instrument(self, model_repository, instrument):
        return model_repository[instrument][0]

    def __put_dynamic_model_for_instrument(self, model_repository, instrument, model, date):
        model_repository[instrument] = (model, date)

    def __get_model_age(self, model_repository, instrument, current_date):
        if instrument not in model_repository.keys():
            return float('inf')

        model_date = self.__get_dynamic_model_date_for_instrument(model_repository, instrument)
        delta = current_date - model_date
        return delta.days



    def shouldBuyStock(self, bar, instrument):
        if self.getBroker().getShares(instrument) > 0:
            return False

        if self.__models is not None and instrument in self.__models.keys():
            return self.__models[instrument].predict(self.generate_predict_dataset(instrument))[0] == 1


        model = None
        current_date = bar.getDateTime()
        model_age = self.__get_model_age(self.__dynamic_buy_models, instrument, current_date)
        if model_age <= self.__max_dynamic_model_age:
            self.__logger.debug("Current date is %s and buy model age for instrument %s is %s, LESS than %s" % (current_date, instrument, model_age, self.__max_dynamic_model_age))
            model = self.__get_dynamic_model_for_instrument(self.__dynamic_buy_models, instrument)
        else:
            self.__logger.debug("Current date is %s and buy model age for instrument %s is %s, MORE than %s" % (current_date, instrument, model_age, self.__max_dynamic_model_age))
            data_set = self.generate_trainingtobuy_dataset(instrument,
                                                           training_window_days=self.__training_buy_window_days,
                                                           forecast_window_days=self.__forecast_buy_window_days)
            X = data_set.drop(labels=['result'], axis=1)
            y = [1 if y else 0 for y in data_set.result]
            if len(set(y)) == 1 or len(X) == 0:
                return False

            model = self.__pipeline.fit(X, y)
            self.__put_dynamic_model_for_instrument(self.__dynamic_buy_models, instrument, model, current_date)


        return model.predict(self.generate_predict_dataset(instrument))[0] == 1

    def shouldSellStock(self, bar, instrument):
        # if bar.getClose() < self.__purchases[instrument]:
        #     return False

        if self.__models is not None and instrument in self.__models.keys():
            return self.__models[instrument].predict(self.generate_predict_dataset(instrument))[0] == 0

        model = None
        current_date = bar.getDateTime()
        model_age = self.__get_model_age(self.__dynamic_sell_models, instrument, current_date)
        if model_age <= self.__max_dynamic_model_age:
            self.__logger.debug("Current date is %s and sell model age for instrument %s is %s, LESS than %s" % (
            current_date, instrument, model_age, self.__max_dynamic_model_age))
            model = self.__get_dynamic_model_for_instrument(self.__dynamic_sell_models, instrument)
        else:
            self.__logger.debug("Current date is %s and sell model age for instrument %s is %s, MORE than %s" % (
            current_date, instrument, model_age, self.__max_dynamic_model_age))
            data_set = self.generate_trainingtosell_dataset(instrument,
                                                            training_window_days=self.__training_sell_window_days,
                                                            forecast_window_days=self.__forecast_sell_window_days)
            X = data_set.drop(labels=['result'], axis=1)
            y = [1 if y else 0 for y in data_set.result]

            if len(X) == 0:
                return False
            if len(set(y)) == 1:
                return y[0] == 1

            model = self.__pipeline.fit(X, y)
            self.__put_dynamic_model_for_instrument(self.__dynamic_sell_models, instrument, model, current_date)

        return model.predict(self.generate_predict_dataset(instrument))[0] == 1

    def calculateEntrySize(self, bar, instrument):

        totalCash = self.getBroker().getEquity()
        closeValue = bar.getClose()
        stopLossPoint = self.calculateStopLoss(bar, instrument)
        # return max(1, math.floor ( (totalCash * self.__riskFactor) / ((closeValue - stopLossPoint) if closeValue!=stopLossPoint else 1) ))
        return min(1000, totalCash) / closeValue

    def calculateStopLoss(self, bar, instrument):
        # low_values = np.array(self._feed[instrument].getLowDataSeries())
        # return min(low_values[-(26 + 2):-2]) if len(low_values) > (26 + 1) else None
        return 0

    def generate_training_dataset(self, instrument, training_window_days, forecast_window_days):
        open_series = self._feed[instrument].getOpenDataSeries()
        close_series = self._feed[instrument].getCloseDataSeries()
        high_series = self._feed[instrument].getHighDataSeries()
        low_series = self._feed[instrument].getLowDataSeries()
        volume_series = self._feed[instrument].getVolumeDataSeries()

        open_values = np.array(open_series) if training_window_days is None else np.array(open_series)[-(
            training_window_days + forecast_window_days):]
        close_values = np.array(close_series) if training_window_days is None else np.array(close_series)[-(
            training_window_days + forecast_window_days):]
        high_values = np.array(high_series) if training_window_days is None else np.array(high_series)[-(
            training_window_days + forecast_window_days):]
        low_values = np.array(low_series) if training_window_days is None else np.array(low_series)[-(
            training_window_days + forecast_window_days):]
        volume_values = np.array(volume_series) if training_window_days is None else np.array(volume_series)[-(
            training_window_days + forecast_window_days):]

        data_set = self.__generate_features(close_values, high_values, low_values, open_values, volume_values, instrument)
        return data_set.fillna(value=0)

    def generate_trainingtobuy_dataset(self, instrument, training_window_days, forecast_window_days):
        data_set = self.generate_training_dataset(instrument=instrument, training_window_days=training_window_days,
                                                  forecast_window_days=forecast_window_days)
        data_set['result'] = self.__buy_result_function(open=data_set.open_price, close=data_set.close_price,
                                                        high=data_set.high_price, low=data_set.low_price,
                                                        volume=data_set.volume_price,
                                                        forecast_window_days=forecast_window_days)
        return data_set[data_set.result.astype(str).ne('None')]

    def generate_trainingtosell_dataset(self, instrument, training_window_days, forecast_window_days):
        data_set = self.generate_training_dataset(instrument=instrument, training_window_days=training_window_days,
                                                  forecast_window_days=forecast_window_days)
        data_set['result'] = self.__sell_result_function(open=data_set.open_price, close=data_set.close_price,
                                                         high=data_set.high_price, low=data_set.low_price,
                                                         volume=data_set.volume_price,
                                                         forecast_window_days=forecast_window_days)
        return data_set[data_set.result.astype(str).ne('None')]

    def generate_predict_dataset(self, instrument):
        open_values = np.array(self._feed[instrument].getOpenDataSeries())
        close_values = np.array(self._feed[instrument].getCloseDataSeries())
        high_values = np.array(self._feed[instrument].getHighDataSeries())
        low_values = np.array(self._feed[instrument].getLowDataSeries())
        volume_values = np.array(self._feed[instrument].getVolumeDataSeries())

        data_set = self.__generate_features(close_values, high_values, low_values, open_values, volume_values, instrument)
        return data_set[-1:].fillna(value=0)

    def __generate_features(self, close_values, high_values, low_values, open_values, volume_values, instrument):
        features = pd.DataFrame()
        features['open_price'] = open_values
        features['close_price'] = close_values
        features['high_price'] = high_values
        features['low_price'] = low_values
        features['volume_price'] = volume_values
        technical_indicators = self.__generate_technical_indicators(close_values, high_values, low_values, open_values,
                                             volume_values)
        for ti in technical_indicators.columns:
            features[ti] = technical_indicators[ti]

        other_instruments =  list(set(self._feed.getRegisteredInstruments()) - set([instrument]))
        for instr in other_instruments:
            open = np.array(self._feed[instr].getOpenDataSeries())
            high = np.array(self._feed[instr].getHighDataSeries())
            low = np.array(self._feed[instr].getLowDataSeries())
            close = np.array(self._feed[instr].getCloseDataSeries())
            volume = np.array(self._feed[instr].getVolumeDataSeries())
            max_size = len(open_values)

            features['open_' + instr] = self.__force_size(list(open), max_size)
            features['open_movement_'+instr] = self.__force_size(list(np.true_divide(np.diff(open), open[:-1])), max_size)
            features['high_' + instr] = self.__force_size(list(high), max_size)
            features['high_movement_' + instr] = self.__force_size(list(np.true_divide(np.diff(high), high[:-1])), max_size)
            features['low_' + instr] = self.__force_size(list(low), max_size)
            features['low_movement_' + instr] = self.__force_size(list(np.true_divide(np.diff(low), low[:-1])), max_size)
            features['close_'+instr] = self.__force_size(list(close), max_size)
            features['close_movement_' + instr] = self.__force_size(list(np.true_divide(np.diff(close), close[:-1])), max_size)

            technical_indicators = self.__generate_technical_indicators(close_values, high_values, low_values,
                                                                        open_values,
                                                                        volume_values)
            for ti in technical_indicators.columns:
                features["%s_%s" % (ti, instr)] = technical_indicators[ti]

        return features

    def __generate_technical_indicators(self, close_values, high_values, low_values, open_values,
                                        volume_values):
        features = pd.DataFrame()
        features['short_sma'] = talib.SMA(close_values, 5)
        features['long_sma'] = talib.SMA(close_values, 20)
        features['sma_diff'] = features.long_sma - features.short_sma
        features['stochf0'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[0]
        features['stochf1'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[1]
        features['rsi'] = talib.RSI(close_values, 20)
        features['ad'] = talib.AD(high=high_values, low=low_values, close=close_values, volume=volume_values)
        features['dema'] = talib.DEMA(close_values)
        features['ema'] = talib.EMA(close_values)
        features['ht_trendiline'] = talib.HT_TRENDLINE(close_values)
        features['kama'] = talib.KAMA(close_values)
        features['midpoint'] = talib.MIDPOINT(close_values)
        features['midprice'] = talib.MIDPRICE(high=high_values, low=low_values)
        features['sar'] = talib.SAR(high=high_values, low=low_values)
        features['sarext'] = talib.SAREXT(high=high_values, low=low_values)
        features['adx'] = talib.ADX(high=high_values, low=low_values, close=close_values)
        features['adxr'] = talib.ADXR(high=high_values, low=low_values, close=close_values)
        features['apo'] = talib.APO(close_values)
        features['aroon0'] = talib.AROON(high=high_values, low=low_values)[0]
        features['aroon1'] = talib.AROON(high=high_values, low=low_values)[1]
        features['aroonosc'] = talib.AROONOSC(high=high_values, low=low_values)
        features['bop'] = talib.BOP(open=open_values, high=high_values, low=low_values, close=close_values)
        features['cmo'] = talib.CMO(close_values)
        features['dx'] = talib.DX(high=high_values, low=low_values, close=close_values)
        features['macdfix0'] = talib.MACDFIX(close_values)[0]
        features['macdfix1'] = talib.MACDFIX(close_values)[1]
        features['macdfix2'] = talib.MACDFIX(close_values)[2]
        features['mfi'] = talib.MFI(high=high_values, low=low_values, close=close_values, volume=volume_values)
        features['minus_di'] = talib.MINUS_DI(high=high_values, low=low_values, close=close_values)
        features['minus_dm'] = talib.MINUS_DM(high=high_values, low=low_values)
        features['mom'] = talib.MOM(close_values)
        features['plus_di'] = talib.PLUS_DI(high=high_values, low=low_values, close=close_values)
        features['plus_dm'] = talib.PLUS_DM(high=high_values, low=low_values)
        features['ppo'] = talib.PPO(close_values)
        features['roc'] = talib.ROC(close_values)
        features['stochf0'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[0]
        features['stochf1'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[1]
        features['stochrsi0'] = talib.STOCHRSI(close_values)[0]
        features['stochrsi1'] = talib.STOCHRSI(close_values)[1]
        # data_set['trix'] = talib.TRIX(close_values)
        features['ultosc'] = talib.ULTOSC(high=high_values, low=low_values, close=close_values)
        features['willr'] = talib.WILLR(high=high_values, low=low_values, close=close_values)
        features['adosc'] = talib.ADOSC(high=high_values, low=low_values, close=close_values, volume=volume_values)
        features['obv'] = talib.OBV(close_values, volume_values)
        features['ht_dcperiod'] = talib.HT_DCPERIOD(close_values)
        features['ht_dcphase'] = talib.HT_DCPHASE(close_values)
        features['ht_phasor0'] = talib.HT_PHASOR(close_values)[0]
        features['ht_phasor1'] = talib.HT_PHASOR(close_values)[1]
        features['ht_sine0'] = talib.HT_SINE(close_values)[0]
        features['ht_sine1'] = talib.HT_SINE(close_values)[1]
        features['ht_trendmode'] = talib.HT_TRENDMODE(close_values)
        features['atr'] = talib.ATR(high=high_values, low=low_values, close=close_values)
        features['trange'] = talib.TRANGE(high=high_values, low=low_values, close=close_values)
        features['CDL2CROWS'] = talib.CDL2CROWS(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDL3INSIDE'] = talib.CDL3INSIDE(open=open_values, high=high_values, low=low_values,
                                                  close=close_values)
        features['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(open=open_values, high=high_values, low=low_values,
                                                              close=close_values)
        features['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(open=open_values, high=high_values, low=low_values,
                                                              close=close_values)
        features['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(open=open_values, high=high_values, low=low_values,
                                                            close=close_values)
        features['CDLBELTHOLD'] = talib.CDLBELTHOLD(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(open=open_values, high=high_values, low=low_values,
                                                      close=close_values)
        features['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(open=open_values, high=high_values, low=low_values,
                                                                    close=close_values)
        features['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(open=open_values, high=high_values, low=low_values,
                                                              close=close_values)
        features['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLDOJI'] = talib.CDLDOJI(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLDOJISTAR'] = talib.CDLDOJISTAR(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open=open_values, high=high_values, low=low_values,
                                                              close=close_values)
        features['CDLENGULFING'] = talib.CDLENGULFING(open=open_values, high=high_values, low=low_values,
                                                      close=close_values)
        features['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(open=open_values, high=high_values, low=low_values,
                                                                    close=close_values)
        features['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLHAMMER'] = talib.CDLHAMMER(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open=open_values, high=high_values, low=low_values,
                                                        close=close_values)
        features['CDLHARAMI'] = talib.CDLHARAMI(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLHIKKAKE'] = talib.CDLHIKKAKE(open=open_values, high=high_values, low=low_values,
                                                  close=close_values)
        features['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(open=open_values, high=high_values, low=low_values,
                                                        close=close_values)
        features['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(open=open_values, high=high_values, low=low_values,
                                                            close=close_values)
        features['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLINNECK'] = talib.CDLINNECK(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLKICKING'] = talib.CDLKICKING(open=open_values, high=high_values, low=low_values,
                                                  close=close_values)
        features['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(open=open_values, high=high_values, low=low_values,
                                                            close=close_values)
        features['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLLONGLINE'] = talib.CDLLONGLINE(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLMARUBOZU'] = talib.CDLMARUBOZU(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLMATHOLD'] = talib.CDLMATHOLD(open=open_values, high=high_values, low=low_values,
                                                  close=close_values)
        features['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLONNECK'] = talib.CDLONNECK(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLPIERCING'] = talib.CDLPIERCING(open=open_values, high=high_values, low=low_values,
                                                    close=close_values)
        features['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(open=open_values, high=high_values, low=low_values,
                                                                    close=close_values)
        features['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open=open_values, high=high_values, low=low_values,
                                                            close=close_values)
        features['CDLSHORTLINE'] = talib.CDLSHORTLINE(open=open_values, high=high_values, low=low_values,
                                                      close=close_values)
        features['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open=open_values, high=high_values, low=low_values,
                                                          close=close_values)
        features['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(open=open_values, high=high_values, low=low_values,
                                                                close=close_values)
        features['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(open=open_values, high=high_values, low=low_values,
                                                              close=close_values)
        features['CDLTAKURI'] = talib.CDLTAKURI(open=open_values, high=high_values, low=low_values, close=close_values)
        features['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(open=open_values, high=high_values, low=low_values,
                                                      close=close_values)
        features['CDLTHRUSTING'] = talib.CDLTHRUSTING(open=open_values, high=high_values, low=low_values,
                                                      close=close_values)
        features['CDLTRISTAR'] = talib.CDLTRISTAR(open=open_values, high=high_values, low=low_values,
                                                  close=close_values)
        features['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(open=open_values, high=high_values, low=low_values,
                                                            close=close_values)
        features['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(open=open_values, high=high_values, low=low_values,
                                                                  close=close_values)
        features['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(open=open_values, high=high_values, low=low_values,
                                                                    close=close_values)
        return features

    def __force_size(self, list_of_values, size, fill_value=0):
        if len(list_of_values)>size:
            return list_of_values[:size]

        if len(list_of_values)<size:
            while len(list_of_values) < size:
                list_of_values.insert(0, fill_value)
            return list_of_values

        return list_of_values


class CorrelationAnalysisTradingAlgorithm(TradingAlgorithm):
    def __init__(self, feed, broker, riskFactor, model):
        super(CorrelationAnalysisTradingAlgorithm, self).__init__(feed, broker)
        self.__riskFactor = riskFactor
        self.__model = model
        self.buyscores = []
        self.sellscores = []

    def shouldAnalyze(self, bar, instrument):
        return len(self._feed[instrument].getCloseDataSeries()) > 30

    def shouldBuyStock(self, bar, instrument):
        movements_frame = self.generate_data_set(instrument)
        score = self.__model.predict(movements_frame)[0]
        self.buyscores.append(score)
        return score > 0.03

    def shouldSellStock(self, bar, instrument):
        movements_frame = self.generate_data_set(instrument)
        score = self.__model.predict(movements_frame)[0]
        self.sellscores.append(score)
        return score < -0.05

    def calculateEntrySize(self, bar, instrument):
        totalCash = self.getBroker().getEquity()
        closeValue = bar.getClose()
        return min(1000, totalCash) / closeValue

    def calculateStopLoss(self, bar, instrument):
        low_values = np.array(self._feed[instrument].getLowDataSeries())
        return min(low_values[-(26 + 2):-2]) if len(low_values) > (26 + 1) else None

    def generate_data_set(self, instrument):
        instruments = self._feed.getRegisteredInstruments()
        movements_frame = pd.DataFrame()
        movements_frame[self.__model.evalcodeparam] = [instrument]
        for code in instruments:
            closevalues = self._feed[code].getCloseDataSeries()
            movements_frame[code] = [(closevalues[-1] - closevalues[-2]) / closevalues[-2]]

        return movements_frame
