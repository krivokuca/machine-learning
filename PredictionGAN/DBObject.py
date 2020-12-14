import pymysql
import uuid
import collections
import time
import os
import datetime
import json
import base64
import pandas as pd
from DBUtils.SteadyDB import connect


class DBObject():
    '''
    The DBObjects is the primary way to insert, get and delete data from the
    database.
    '''

    def __init__(self):
        self.credentials = ['localhost', 'username', 'password', 'database']
        self.db = connect(
            creator=pymysql,
            host=self.credentials[0],
            user=self.credentials[1],
            password=self.credentials[2],
            database=self.credentials[3],
            autocommit=True,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def get_stock_data(self, ticker, limit=False, period=False):
        '''
        Retrieves all the stock data pertaining to
        the stock. If full is False, the first 100
        rows are returned. Ticker can be either 1 ticker 
        value or an array of ticker values.
        '''
        cursor = self.db.cursor()
        select_query = "SELECT `date`, `open`, `high`, `low`, `close`, `volume` FROM `stock_data` WHERE `ticker` = %s"
        if period and type(period) == list:
            select_query += " AND (`date` BETWEEN '{}' AND '{}')".format(
                period[0], period[1])
        elif period and type(period) == str or isinstance(period, datetime.date):
            select_query += " AND `date` = '{}'".format(period)
        select_query += ' ORDER BY `date` DESC'
        lim = " LIMIT 100;" if not limit else ';' if limit == 'false' else ' LIMIT {};'.format(
            int(limit))
        select_query += lim
        # single ticker value
        if(type(ticker) is str):
            cursor.execute(select_query, ticker)
            results = cursor.fetchall()
            if results:
                return {
                    ticker: results[::-1]
                }
            return False
        else:
            # multiple ticker values
            outdict = {}
            for tick in ticker:
                cursor.execute(select_query, tick)
                results = cursor.fetchall()
                outdict[tick] = results[::-1]
            return outdict
