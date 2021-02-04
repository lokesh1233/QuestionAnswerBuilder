from config import configuration as config
import psycopg2
from .util import utility


class datasetQA():
    def __init__(self):
        self.psdatabase = utility()
        pass

    def load_dataset(self):
        dbQuery = f"select question, answer from {config.qabotTable} WHERE update_ind IS NOT NULL"
        return self.psdatabase.connect_execute(dbQuery)


