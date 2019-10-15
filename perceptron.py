
class Perceptron:

    def __init__(self, weights, threshold, weights_learning_rate, threshold_learning_rate):
        self.weights = weights
        self.threshold = threshold
        self.weights_learning_rate = weights_learning_rate
        self.threshold_learning_rate = threshold_learning_rate

    def predict(self, to_predict):
        return 1 if sum(x * y for x, y in zip(self.weights, to_predict)) >= self.threshold else 0

    def predict_raw(self, to_predict):
        return sum(x * y for x, y in zip(self.weights, to_predict)) / self.threshold

    def train(self, to_predict, expected_answer):
        answer = self.predict(to_predict)
        if answer != expected_answer:
            self.threshold += (answer - expected_answer) * self.threshold_learning_rate
            multiplier = (expected_answer - answer) * self.weights_learning_rate
            self.weights = [a + b for a, b in zip(self.weights, [x * multiplier for x in to_predict])]

    def train_df(self, df):
        for index, row in df.iterrows():
            self.train(row.values[0:-1], row.values[-1])

    def check_accuracy(self, df):
        i = 0
        n = 0
        for index, row in df.iterrows():
            if self.predict(row.values[0:-1]) == row.values[-1]:
                i += 1
            n += 1
        print("{}/{}".format(i, n))
