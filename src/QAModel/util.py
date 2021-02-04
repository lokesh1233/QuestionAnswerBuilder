import psycopg2
from config import configuration as config
import requests
import json
from threading import Thread
import time
# from grequests import async as sync
class utility():

    def __init__(self):
        self._db_connection()


    def _db_connection(self):
        # Set up a connection to the postgres server.
        try:
            conn_string = "host=" + config.postgreSql_dbhost + " port=" + config.postgreSql_dbport + " dbname=" + config.postgreSql_dbname + " user=" + config.postgreSql_dbuser \
                          + " password=" + config.postgreSql_dbpswd
            self._connection = psycopg2.connect(conn_string)
            self._connection.autocommit = True
            # self._connectionCursor = conn.cursor()
        except Exception as err:
            print(f'Connection err: {err}')



    def connect_execute(self, dbQuery, params = None):
        # Set up a connection to the postgres server.
        # conn_string = "host=" + config.postgreSql_dbhost + " port=" + config.postgreSql_dbport + " dbname=" + config.postgreSql_dbname + " user=" + config.postgreSql_dbuser \
        #               + " password=" + config.postgreSql_dbpswd
        if self._connection.closed > 0:
            self._db_connection()

        result = None
        try:
            # with psycopg2.connect(conn_string) as connection:
            #     cursor = connection.cursor()
            with self._connection, self._connection.cursor() as cur:
                cur.execute(dbQuery, params)
                self._connection.commit()
                result = self.handleResponses(cur)
        except psycopg2.Error as error:
            print(f'error while connect to PostgreSQL : '
                  f'{error}')
            result = self.handleErrorResponse(error)
        finally:
            # if cursor:
            #     cursor.close()
                # connection.close()
            # print('PostgreSQL connection to is closed')
            return result
        return result

    def handleResponses(self, cursor):
        try:
            result = [dict((cursor.description[i][0], value) for i, value in enumerate(row)) for row in cursor.fetchall()]
            return result;
        except:
            return {"message":cursor.statusmessage,"type":"S"}

    def handleErrorResponse(self, res):
        try:
            return {"message":f'{res}'.split("DETAIL: ")[1],
             "type":"E"}
        except:
            return {"message": f'{res}',
             "type": "E"}


    def QABotDataRequest(self, **data):
        time.sleep(1)
        requests.post(config.insertQABot, data=json.dumps(data), headers={"Content-Type": "application/json"})


    async def insertQABotData(self, data):

        thread = Thread(target=self.QABotDataRequest, kwargs=data)
        thread.start()
        return ""



