import tensorflow as tf
assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
import glob
import os

from sanic import response
import pickle
from src.QAModel.transformerQA import TransformerQA
from .QAModel.util import utility
from config import configuration
import tensorflow_datasets as tfds
import os
import re
import logging
# import numpy as np
# import pandas as pd
#
# import matplotlib.pyplot as plt

class transformerChat():
    # Hyper-parameters
    NUM_LAYERS = 2
    D_MODEL = 128
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1
    MAX_LENGTH = 40

    def __init__(self):
        self.transQA = TransformerQA()
        self.utilityDB = utility()

    def buildModel(self):
        self.initilizeTokenizer()
        self.model = self.create_model(self.VOCAB_SIZE)
        self.model.load_weights(configuration.modelPath)


    def initilizeTokenizer(self):
        # loading
        with open(configuration.tokenizerPath, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Define start and end token to indicate the start and end of a sentence
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]

        # Vocabulary size plus start and end token
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2


    def create_model(self, VOCAB_SIZE):
        tf.keras.backend.clear_session()
        model = self.transQA.transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=self.NUM_LAYERS,
            units=self.UNITS,
            d_model=self.D_MODEL,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT)
        return model

    async def modelList(self):
        dirs = glob.glob("models/*")
        modelsList = []
        for dir in dirs:
            if os.path.isdir(dir):
                modelId = os.path.split(dir)[1]
                modelsList.append({
                    "model":modelId,
                    "comment":"",
                    "isActive": self.isModelActivate(modelId)
                })
        return modelsList

    def isModelActivate(self, modelId):
        return modelId == configuration.modelPath.split("/")[1]

    async def deleteModel(self, modelId):
        if self.isModelActivate(modelId):
            return {"message":f"Current model {modelId} is active", "type":"E"}

        dirs = glob.glob(f"models/{modelId}/*")
        for filePth in dirs:
            if os.path.isfile(filePth):
                os.remove(filePth)
        os.removedirs(f"models/{modelId}")
        return {"message":f"Model {modelId} is deleted", "type":"s"}

    async def predict(self, req):
        sentence = req['message']
        sender = req.get("sender", None)
        prediction = self.evaluate(sentence)
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])
        logging.warning('Input: {}'.format(sentence))
        logging.warning('Output: {}'.format(predicted_sentence))
        dta = {
            "question":sentence,
            "answer":predicted_sentence,
            "user_id":sender
        }
        await self.utilityDB.insertQABotData(dta)
        return {"text":predicted_sentence}

    def evaluate(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        sentence = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)
        output = tf.expand_dims(self.START_TOKEN, 0)

        # removing loop condition
        # predictions = self.model(inputs=[sentence, output], training=False)
        #
        # # select the last word from the seq_len dimension
        # predictions = predictions[:, -1:, :]
        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        #
        # # return the result if the predicted_id is equal to the end token
        # if not tf.equal(predicted_id, self.END_TOKEN[0]):
        #     output = tf.concat([output, predicted_id], axis=-1)

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        # output = tf.concat([output, predicted_id], axis=-1)

        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

            # print(predictions[0][0][predicted_id[0][0].numpy()].numpy())

            if predictions[0][0][predicted_id[0][0].numpy()].numpy() < 5.0:
                output = tf.expand_dims(self.START_TOKEN, 0)
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)



    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence