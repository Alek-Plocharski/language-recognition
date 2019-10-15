from perceptron import Perceptron
from string import ascii_lowercase
from collections import Counter
import random
import pandas as pd
import os


def count_letters_and_normalize(file_path):
    with open(file_path) as file:
        counter = Counter(letter for line in file for letter in line.lower() if letter in ascii_lowercase)
        result = []
        for letter in ascii_lowercase:
            result.append(counter[letter])
        result_sum = sum(result) if sum(result) > 0 else 1
        return [float(i) / result_sum for i in result]


class LanguageClassifier:

    def __init__(self, languages, weights_learning_rate, threshold_learning_rate):
        self.languages = languages
        self.perceptrons = []
        for language in languages:
            self.perceptrons.append(Perceptron([random.uniform(0, 1) for _ in ascii_lowercase],
                                               random.uniform(0, 1),
                                               weights_learning_rate,
                                               threshold_learning_rate))

    def train(self):
        path = "./training"
        columns = list(ascii_lowercase)
        columns.append('decision')
        for language in self.languages:
            counts_list = []
            dir_path = path + "/" + language
            directory = os.fsencode(dir_path)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                counts = count_letters_and_normalize(dir_path + "/" + filename)
                counts.append(0)
                counts_list.append(counts)
            df = pd.DataFrame(counts_list, columns=columns)
            for i, perceptron in enumerate(self.perceptrons):
                df['decision'] = 1 if self.languages[i] == language else 0
                perceptron.train_df(df)

    def predict(self, observation):
        predictions = [perceptron.predict_raw(observation) for perceptron in self.perceptrons]
        return languages[predictions.index(max(predictions))]

    def test(self):
        path = "./test"
        i = 0
        n = 0
        for language in self.languages:
            dir_path = path + "/" + language
            directory = os.fsencode(dir_path)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                v = count_letters_and_normalize(dir_path + "/" + filename)
                prediction = self.predict(v)
                if prediction == language:
                    i += 1
                n += 1
        print("{}/{}".format(i, n))


languages = ['eng', 'es', 'du']
model = LanguageClassifier(languages, 0.05, 0.05)
for _ in range(500):
    model.train()
model.test()
