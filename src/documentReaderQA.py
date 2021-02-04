import datetime
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from src.documentReader import DocumentReader
import wikipedia as wiki

from transformers import pipeline

class documentReaderQA:
    def __init__(self):
        self.loadModel()


    def loadModel(self):
        # executing these commands for the first time initiates a download of the
        # model weights to ~/.cache/torch/transformers/
        # self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        # self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
        self.reader = DocumentReader("distilbert-base-uncased-distilled-squad")
        # self.nlpQA = pipeline('question-answering')

        # tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        # model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

        # self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        # self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    def predictAnswer(self, question):
        question = "Who ruled Macedonia"

        context = """Macedonia was an ancient kingdom on the periphery of Archaic and Classical Greece, 
        and later the dominant state of Hellenistic Greece. The kingdom was founded and initially ruled 
        by the Argead dynasty, followed by the Antipatrid and Antigonid dynasties. Home to the ancient 
        Macedonians, it originated on the northeastern part of the Greek peninsula. Before the 4th 
        century BC, it was a small kingdom outside of the area dominated by the city-states of Athens, 
        Sparta and Thebes, and briefly subordinate to Achaemenid Persia."""

        # 1. TOKENIZE THE INPUT
        # note: if you don't include return_tensors='pt' you'll get a list of lists which is easier for
        # exploration but you cannot feed that into a model.
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt")

        # 2. OBTAIN MODEL SCORES
        # the AutoModelForQuestionAnswering class includes a span predictor on top of the model.
        # the model returns answer start and end scores for each word in the text
        answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start = torch.argmax(
            answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

        # 3. GET THE ANSWER SPAN
        # once we have the most likely start and end tokens, we grab all the tokens between them
        # and convert tokens back to words!
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    def predictWikiAnswers(self, question):
        # questions = [
        #     'When was Barack Obama born?',
        #     'Why is the sky blue?',
        #     'How many sides does a pentagon have?'
        # ]

        # question = "How many sides does a pentagon have"

        # reader = DocumentReader("deepset/bert-base-cased-squad2")

        # if you trained your own model using the training cell earlier, you can access it with this:
        # reader = DocumentReader("./models/bert/bbu_squad2")

        for question in [question]:
            print(f"Question: {question}")
            results = wiki.search(question, results=1)

            if len(results) == 0:
                return ""

            text = wiki.page(results[0]).summary[:800].lower()
            # print(f"Top wiki result: {page}")

            # text = page.content
            print(text)

            self.reader.tokenize(question, text)
            answer = self.reader.get_answer()
            print(f"Answer: {answer}")
            print()
            return answer

            # result = self.nlpQA({
            #     'question': question,
            #     'context': text
            # })
            # print(result)
            # return result




