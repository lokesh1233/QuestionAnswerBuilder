from .load_dataSet import datasetQA
from .preprocessing import preprocessing
from .optimizer import Optimizer
from ..transformer import transformerChat
from sanic import response
import time
import threading
import os
import logging as log
import pickle
from .util import utility

class transformerTrain():
    def __init__(self):
        self.EPOCH = 200
        self.dataset = datasetQA()
        self.preprocessingQA = preprocessing()
        self.ADOptimizer = Optimizer()
        self.transformerModel=transformerChat()
        self.postGresSqlDB = utility()

    def timeStampModelFolder(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.saveTimeStr = str(timestr)
        self.createModelPath = "models/"+str(timestr)
        os.makedirs(self.createModelPath)


    async def safeLoaderTimerTrain(self):
        res, stamp =await self.isModelRunning()
        if type(res) == str:
            return {"message": res, "type": "S"}
        self.timeStampModelFolder()
        thrd = threading.Thread(target=self.datasetPreprocess_train, args=(self.saveTimeStr,))
        thrd.start()
        # await self.datasetPreprocess_train()
        return {"message": f"Model started training with time stamp {self.saveTimeStr}", "type": "S"}


    def datasetPreprocess_train(self, saveTimeStr):
        try:
            self.saveTimeStr = saveTimeStr
            self.loggerTrainModel("Loading dataset to train a model")
            qaResult = self.dataset.load_dataset()
            if (type(qaResult) == list and len(qaResult)>0) == False:
                self.loggerTrainModel("No dataset found to train a model", "E")
                self.fallBackModelStr()
                return {"message": "No dataset found to train a model", "type": "E"}
            self.loggerTrainModel("Number of records loaded in dataset : " + str(len(qaResult)))
            self.trainset = self.preprocessingQA.preprocess_questions(qaResult)
            self.loggerTrainModel("dataset is preprocessing for questions and answers")
            optimizer, loss_function, accuracy  = self.ADOptimizer.AdamOptimizer()
            self.loggerTrainModel("transformer model is creating")
            self.model = self.transformerModel.create_model(self.preprocessingQA.VOCAB_SIZE)
            self.loggerTrainModel("transformer model is compiling")
            self.model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
            return self.train()
        except Exception as err:
            self.loggerTrainModel(f'{err}', "E")
            self.fallBackModelStr()

    def fallBackModelStr(self):
        self.transformerModel.deleteModel(self.saveTimeStr)
        self.loggerTrainModel("done", "E")


    def loggerTrainModel(self, message, type="S"):
        log.warning(message)
        executeQuery = f"""INSERT INTO parameter(id, type, type2, type3, type4) VALUES('model', 'modelTraining', '{message}', '{type}', '{self.saveTimeStr}')
                    ON CONFLICT(id, type) DO UPDATE SET type2 = excluded.type2, type3 = excluded.type3, type4 = excluded.type4""";
        self.postGresSqlDB.connect_execute(executeQuery)

    def train(self):
        self.loggerTrainModel("transformer model started training for number of epochs : "+ str(self.EPOCH))
        self.model.fit(self.trainset, epochs=self.EPOCH)
        self.loggerTrainModel("transformer model training completed")
        return self.saveModel()

    def saveModel(self):
        # self.timeStampModelFolder()
        self.loggerTrainModel("model saving with: "+str(self.createModelPath.split("/")[1]))
        # saving
        tokenPath = self.createModelPath + '/tokenizer.pickle'
        with open(tokenPath, 'wb') as handle:
            pickle.dump(self.preprocessingQA.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.loggerTrainModel("saved vocabilary in " +tokenPath)
        modelPath = self.createModelPath + "/my_checkpoint"
        self.model.save_weights(modelPath)
        # self.loggerTrainModel("saved model directory in " + modelPath)
        self.loggerTrainModel("done", "S")
        return {"message":"model is trained and saved in "+self.saveTimeStr, "type":"S"}


    async def isModelRunning(self):
        log.warning("identify model is running?")
        executeQuery = f"""SELECT  type2, type3, type4 from parameter where id = 'model' and type='modelTraining'""";
        res = self.postGresSqlDB.connect_execute(executeQuery)
        if type(res) == list and len(res) > 0 and res[0].get('type2', 'done') != 'done' and res[0].get('type3', 'S') == 'S':
            return (res[0].get('type2', 'model is running'), res[0].get('type4', None))
        else:
            return (False, None)

    async def filterRunningModel(self, modelList):
        res, stamp = False, None
        if len(modelList) > 0:
            res, stamp =await self.isModelRunning()
            if res == False:
                return modelList
        for mdl in modelList:
            if mdl.get('model', None) == stamp:
                mdl['comment'] = res
                break
        return modelList





