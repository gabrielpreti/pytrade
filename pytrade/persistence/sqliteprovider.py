from pytrade.persistence import DataProvider
from pyalgotrade.broker import Order
from pyalgotrade import broker
from pyalgotrade.broker import backtesting
import sqlite3
from pyalgotrade.utils import dt

class SQLiteDataProvider(DataProvider):
    __USERS_TABLE = "user"
    __ORDERS_TABLE = "stock_order"
    __SHARES_TABLE = "stock_share"

    __CREATE_USER_TABLE_SQL = """
        create table user (
            user_id integer primary key autoincrement,
            name text not null,
            cash real not null
            )
    """

    __CREATE_SHARES_TABLE_SQL = """
        create table stock_share (
            share_id integer primary key autoincrement,
            user_id integer not null references user(user_id),
            code text not null,
            quantity real not null
        )
    """

    __CREATE_ORDERS_TABLE_SQL = """
    create table stock_order (
        order_id integer primary key autoincrement,
        user_id integer not null references user(user_id),
        order_app_id integer not null,
        action_id integer not null,
        state_id integer not null,
        type_id integer not null,
        on_close integer not null default 0,
        all_or_none integer not null default 0,
        avg_fill_price real,
        commissions real,
        filled integer not null default 0,
        good_till_cancelled integer not null default 0,
        instrument text not null,
        quantity real not null,
        submit_date_time integer,
        stop_loss_value real,
        stop_hit integer not null default 0
	)
    """

    __LOAD_CASH_SQL = "select cash from user where name=?"
    __LOAD_SHARES_SQL = "select ss.code, ss.quantity from stock_share ss inner join user u on u.user_id=ss.user_id where u.name=?"
    __LOAD_ORDERS_SQL = """
        SELECT
            so.order_app_id,
            so.action_id,
            so.state_id,
            so.type_id,
            so.on_close,
            so.all_or_none,
            so.avg_fill_price,
            so.commissions,
            so.filled,
            so.good_till_cancelled,
            so.instrument,
            so.quantity,
            so.submit_date_time,
            so.stop_loss_value,
            so.stop_hit
        from
            stock_order so
            inner join user u on u.user_id=so.user_id
        where
            u.name=?
        """

    __GET_USER_ID_SQL = "select user_id from user where name=?"
    __SAVE_CASH_SQL = "update user set cash=? where name=?"
    __SAVE_SHARES_SQL = "insert into stock_share (user_id, code, quantity) values (?, ?, ?)"
    __DELETE_SHARES_SQL = "delete from stock_share where user_id=?"
    __SAVE_ORDERS_SQL = """
        insert into stock_order
        (user_id, order_app_id, action_id, state_id, type_id, all_or_none, avg_fill_price, commissions, filled, good_till_cancelled, instrument, quantity, submit_date_time, on_close, stop_loss_value, stop_hit)
        values
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    __DELETE_ORDERS_SQL = "delete from stock_order where user_id=?"

    __INSERT_USER_SQL = "insert into user (name, cash) values (?, ?)"
    __SELECT_USER_SQL = "select * from USER where name = ?"
    __DELETE_USER_BY_ID_SQL = "delete from user where user_id = ?"


    def __init__(self, dbFilePath):
        super(SQLiteDataProvider, self).__init__()
        self.__connection = sqlite3.connect(dbFilePath)
        self.__connection.isolation_level = None  # To do auto-commit

    def schemaExists(self):
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"

        cursor = self.__connection.cursor()
        cursor.execute(sql, [self.__USERS_TABLE])

        if cursor.fetchone() is None:
            cursor.close()
            return False

        cursor.execute(sql, [self.__ORDERS_TABLE])
        if cursor.fetchone() is None:
            cursor.close()
            return False

        cursor.execute(sql, [self.__SHARES_TABLE])
        if cursor.fetchone() is None:
            cursor.close()
            return False

        cursor.close()
        return True

    def createSchema(self):
        cursor = self.__connection.cursor()

        dropTableSql = "drop table if exists %s"
        cursor.execute(dropTableSql % (self.__ORDERS_TABLE))
        cursor.execute(dropTableSql % (self.__SHARES_TABLE))
        cursor.execute(dropTableSql % (self.__USERS_TABLE))

        cursor.execute(self.__CREATE_USER_TABLE_SQL)
        cursor.execute(self.__CREATE_SHARES_TABLE_SQL)
        cursor.execute(self.__CREATE_ORDERS_TABLE_SQL)

    def initializeUser(self, username, cash):
        cursor = self.__connection.cursor()
        cursor.execute(self.__INSERT_USER_SQL, [username, cash])
        cursor.close()

    def reinitializeUser(self, username, cash):
        if self.userExists(username):
            userId = self.getUserId(username)
            cursor = self.__connection.cursor()
            cursor.execute(self.__DELETE_SHARES_SQL, [userId])
            cursor.execute(self.__DELETE_ORDERS_SQL, [userId])
            cursor.execute(self.__DELETE_USER_BY_ID_SQL, [userId])

        self.initializeUser(username, cash)

    def userExists(self, username):
        cursor = self.__connection.cursor()
        cursor.execute(self.__SELECT_USER_SQL, [username])
        ret = True if cursor.fetchone() else False
        cursor.close()
        return ret

    def getUserId(self, username):
        cursor = self.__connection.cursor()
        cursor.execute(self.__GET_USER_ID_SQL, [username])
        ret = cursor.fetchone()
        cursor.close()
        return ret[0]

    def __createOrder(self, orderId, actionId, stateId, typeId, onClose, allOrNone, avgFillPrice, commissions, filled, gootTillCancelled, instrument, quantity, submitDateTime, stopLossValue, stopHit):
        action = Order.Action.fromInteger(actionId)
        state = Order.State.fromInteger(stateId)
        orderType = Order.Type.fromInteger(typeId)
        allOrNone = False if (allOrNone==0 or allOrNone is None) else True
        filled = False if (filled == 0 or filled is None) else True
        gootTillCancelled = False if (gootTillCancelled == 0 or gootTillCancelled is None) else True
        stopHit = False if (stopHit == 0 or stopHit is None) else True

        if orderType == Order.Type.STOP:
            order = backtesting.StopOrder(action, instrument, stopLossValue, quantity, broker.IntegerTraits())
            order.setStopHit(stopHit)
        elif orderType == Order.Type.MARKET:
            order = backtesting.MarketOrder(action, instrument, quantity, onClose, broker.IntegerTraits())
            order.stopLossValue = stopLossValue
        else:
            raise("Unsupported order type")
        order.setState(state)
        order.setAllOrNone(allOrNone)
        order.setAvgFillPrice(avgFillPrice)
        order.setCommissions(commissions)
        order.setFilled(filled)
        order.setGoodTillCanceled(gootTillCancelled)
        order.setSubmitted(orderId, submitDateTime)

        return order

    def loadCash(self, username):
        cursor = self.__connection.cursor()
        cursor.execute(self.__LOAD_CASH_SQL, [username])
        ret = cursor.fetchone()
        cursor.close()
        return ret[0]

    def loadOrders(self, username):
        cursor = self.__connection.cursor()
        cursor.execute(self.__LOAD_ORDERS_SQL, [username])
        ret = {}
        for row in cursor:
            ret[row[0]]=self.__createOrder(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], dt.timestamp_to_datetime(row[12]), row[13], row[14])
        cursor.close()
        return ret

    def loadShares(self, username):
        cursor = self.__connection.cursor()
        cursor.execute(self.__LOAD_SHARES_SQL, [username])
        ret = {}
        for row in cursor:
            ret[row[0]] = row[1]
        cursor.close()
        return ret

    def persistCash(self, username, cash):
        cursor = self.__connection.cursor()
        cursor.execute(self.__SAVE_CASH_SQL, [cash, username])
        cursor.close()

    def persistOrders(self, username, orders):
        userId = self.getUserId(username)

        cursor = self.__connection.cursor()
        cursor.execute(self.__DELETE_ORDERS_SQL, [userId])
        for o in orders.values():
            parameters = [userId,
                          o.getId(),
                          o.getAction(),
                          o.getState(),
                          o.getType(),
                          o.getAllOrNone(),
                          o.getAvgFillPrice(),
                          o.getCommissions(),
                          o.getFilled(),
                          o.getGoodTillCanceled(),
                          o.getInstrument(),
                          o.getQuantity(),
                          dt.datetime_to_timestamp(o.getSubmitDateTime())]
            if o.getType() == Order.Type.STOP:
                parameters.append(0)
                parameters.append(o.getStopPrice())
                parameters.append(o.getStopHit())
            elif o.getType() == Order.Type.MARKET:
                parameters.append(o.getFillOnClose())
                parameters.append(o.stopLossValue if o.getAction() == Order.Action.BUY else 0)
                parameters.append(0)
            else:
                raise ("Unsupported order type")
            cursor.execute(self.__SAVE_ORDERS_SQL, parameters)
        cursor.close()

    def persistShares(self, username, shares):
        userId = self.getUserId(username)

        cursor = self.__connection.cursor()
        cursor.execute(self.__DELETE_SHARES_SQL, [userId])
        for code, quantity in shares.iteritems():
            cursor.execute(self.__SAVE_SHARES_SQL, [userId, code, quantity])
        cursor.close()

    def getLastValuesForInstrument(self, instrument, date):
        sql = "select b.timestamp, b.open, b.high, b.low, b.close, b.volume from bar b inner join instrument i on i.instrument_id=b.instrument_id where i.name=? and b.timestamp<=? order by b.timestamp desc"
        cursor = self.__connection.cursor()
        cursor.execute(sql, [instrument, dt.datetime_to_timestamp(date)])
        ret = cursor.next()
        return (dt.timestamp_to_datetime(ret[0]), ret[1], ret[2], ret[3], ret[4], ret[5])