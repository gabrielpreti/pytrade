import unittest
from pyalgotrade.tools import googlefinance
from pytrade.feed import DynamicFeed
import pytz, datetime


class TestDynamicFeed(unittest.TestCase):
    dynamicFeed = None

    @classmethod
    def setUpClass(cls):
        rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or \
                                row["Volume"] == "-"
        instruments = ["PETR4", "PETR3"]
        googleFeed = googlefinance.build_feed(instruments, 2015, 2015, storage="./googlefinance", skipErrors=True,
                                              rowFilter=rowFilter)
        cls.dynamicFeed = DynamicFeed("./sqlitedb", instruments, maxLen=10)
        cls.dynamicFeed.getDatabase().addBarsFromFeed(googleFeed)

    def testFirstFeedDay(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 1, tzinfo=pytz.UTC))

        self.assertIsNone(self.dynamicFeed.getCurrentDateTime())
        self.assertIsNone(self.dynamicFeed.getCurrentBars())
        self.assertEqual(len(self.dynamicFeed.getDataSeries("PETR4")), 0)
        self.assertEqual(len(self.dynamicFeed.getDataSeries("PETR3")), 0)

    def testFeedAt20150102(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.36)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.00)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00])

    def testFeedAt20150103(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 3, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.36)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.00)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00])

    def testFeedAt20150104(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 4, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.36)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20],  [9.36])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.00)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00])

    def testFeedAt20150105(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 5, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 5, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 8.61)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.27)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27])

    def testFeedAt20150106(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 6, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 6, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 8.33)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.06)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06])

    def testFeedAt20150107(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 7, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 7, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 8.67)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.45)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45])

    def testFeedAt20150108(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 8, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 8, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.18)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.02)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02])

    def testFeedAt20150109(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.40)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18, 9.4])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.29)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02, 9.29])

    def testFeedAt20150110(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 10, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.40)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18, 9.4])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.29)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02, 9.29])

    def testFeedAt20150111(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 11, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.40)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18, 9.4])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.29)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02, 9.29])

    def testFeedAt20150112(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 12, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 12, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 8.91)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.77)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77])

    def testFeedAt20150113(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 13, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 13, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.00)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91, 9.00])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.83)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83])

    def testFeedAt20150116(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 1, 16, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 1, 16, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 9.44)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20], [8.61, 8.33, 8.67, 9.18, 9.40, 8.91, 9.00, 8.74, 9.34, 9.44])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 9.23)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20], [8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83, 8.50, 9.25,9.23])

    def testFeedAt20151230(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 6.70)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20],
                              [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.57)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20],
                              [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57])

    def testFeedAt20151231(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2015, 12, 31, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 6.70)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20],
                              [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.57)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20],
                              [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57])

    def testFeedAt20160101(self):
        self.dynamicFeed.positionFeed(datetime.datetime(2016, 1, 1, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentDateTime(), datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC))

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR4").getClose(), 6.70)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20],
                              [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70])

        self.assertEqual(self.dynamicFeed.getCurrentBars().getBar("PETR3").getClose(), 8.57)
        self.assertItemsEqual(self.dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20],
                              [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57])

    def test_upper(self):
        print "test_upper"
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        print "test_isupper"
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        print "test_split"
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
