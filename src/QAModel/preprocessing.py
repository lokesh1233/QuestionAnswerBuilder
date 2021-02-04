import re
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf

class preprocessing():
    def __init__(self):
        self.MAX_LENGTH = 40
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 200

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

    def preprocess_questions(self, dataset):
        df = pd.DataFrame(dataset)
        df['question'] = df['question'].apply(self.preprocess_sentence)
        df['answer'] = df['answer'].apply(self.preprocess_sentence)
        questions, answers = df['question'].values.tolist(), df['answer'].values.tolist()
        return self.tokenizersUsingTfds(questions, answers)

    def tokenizersUsingTfds(self, questions, answers):
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2 ** 13)
        self.START_TOKEN, self.END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = tokenizer.vocab_size + 2
        self.tokenizer = tokenizer
        questions, answers = self.tokenize_and_filter(questions, answers)

        # decoder inputs use the previous target as input
        # remove START_TOKEN from targets
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answers[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            },
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs):
        # Maximum sentence length
        MAX_LENGTH = self.MAX_LENGTH
        tokenizer = self.tokenizer
        START_TOKEN = self.START_TOKEN
        END_TOKEN = self.END_TOKEN
        tokenized_inputs, tokenized_outputs = [], []
        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs



