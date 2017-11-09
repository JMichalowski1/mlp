import random
import numpy as np


class MLPModel:

    def __init__(self, number_of_hidden_units=512, learning_rate=0.01, batch_size=1, epochs=100, number_of_classes=10, input_shape=70):
        self.number_of_hidden_units = number_of_hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def trainModel(self, train_x, train_y):
        hidden_weights, output_weights = self.initialize_weights()
        dataset = self.make_tuples(train_x, train_y)
        for i in range(self.epochs):
            random.shuffle(dataset)
            for bundle in dataset:
                sample = bundle[0]
                label = bundle[1]

                # feed forward
                hidden_actual_input = self.multiply(hidden_weights, sample, True)
                hidden_output_vector = self.sigmoid(hidden_actual_input)
                output_actual_input = self.multiply(self.transpose_matrix(output_weights), hidden_output_vector, False)
                output_vector = self.sigmoid(output_actual_input)
                error = self.substract_vectors(output_vector, label)
                print "Error: " + str(np.mean(np.abs(error)))

                # back propagation
                output_delta = self.d_sigmoid(self.hadamard_vector(output_actual_input, error))
                hidden_delta = self.d_sigmoid(self.hadamard_matrix(hidden_output_vector, self.multiply(output_weights, output_delta, True)))

                output_weights = self.substract(output_weights, self.mul_by_skalar(self.transpose_matrix(self.multiply(output_delta, self.transpose(hidden_output_vector), False)), self.learning_rate))
                hidden_weights = self.substract(hidden_weights, self.mul_by_skalar(self.multiply(hidden_delta, self.transpose(sample), False), self.learning_rate))

            print "current epoch: {}".format(i)

        return hidden_weights, output_weights, 100
        #TODO: rysowanie wykresu

    def initialize_weights(self):
        hidden_weights = np.zeros((self.number_of_hidden_units, 70), dtype=float)
        output_weights = np.zeros((self.number_of_hidden_units, self.number_of_classes), dtype=float)
        for i in range(len(hidden_weights)):
            for j in range(70):
                hidden_weights[i][j] = random.uniform(-1, 1)
        for i in range(len(output_weights)):
            for j in range(self.number_of_classes):
                output_weights[i][j] = random.uniform(-1, 1)
        return hidden_weights, output_weights

    def d_sigmoid(self, x):
        return (1 + np.exp(-x)) * (1 - (1/(1 + np.exp(-x))))


    @staticmethod
    def substract(matrix1, matrix2):
        return np.asarray([[a-b for a, b in zip(xrow, yrow)] for xrow, yrow in zip(matrix1, matrix2)])

    @staticmethod
    def multiply(X, Y, isVector):
        if not isVector:
            output = MLPModel.mul_matrix(X, Y)
        else:
            output = MLPModel.multiply_when_vec(X, Y)
        return output

    @staticmethod
    def mul_matrix(X, Y):
        Xx = X.shape[0]
        Yy = Y.shape[1]
        wsp = X.shape[1]
        output = np.zeros((Xx, Yy), dtype=float)
        for i in range(Xx):
            for j in range(Yy):
                sum = 0
                for k in range(wsp):
                    sum += X[i, k] * Y[k, j]
                output[i, j] = sum
        return output

    @staticmethod
    def multiply_when_vec(X, Y):
        Xx = X.shape[0]
        Yy = Y.shape[0]
        wsp = X.shape[1]
        output = np.zeros((Xx, 1), dtype=float)
        for i in range(Xx):
            sum = 0
            for k in range(wsp):
                sum += X[i, k]*Y[k]
            output[i, 0] = sum
        return output

    @staticmethod
    def multiply_matrix(X, Y):
        x_size = X.shape[1]
        y_size = Y.shape[0]
        output = np.zeros((x_size, y_size), dtype=float)
        for i in range(x_size):
            for j in range(y_size):
                output[i][j] = X[i]*Y[j]
        return output

    @staticmethod
    def sum_rows(matrix):
        x_size = matrix.shape[0]
        y_size = matrix.shape[1]
        output = np.zeros((x_size, 1), dtype=float)
        for i in range(x_size):
            sum = 0
            for j in range(y_size):
                sum += matrix[i][j]
            output[i] = sum
        return output

    @staticmethod
    def substract_vectors(X, Y):
        x_size = X.shape[0]
        output = np.zeros((x_size, 1), dtype=float)
        for i in range(x_size):
            output[i] = X[i] - Y[i]
        return output


    @staticmethod
    def multiply_vec(vec1, vec2):
        return np.asarray([el1*el2 for el1, el2 in zip(vec1, vec2)])

    @staticmethod
    def transpose(X):
        output = np.zeros((1, X.shape[0]), dtype=float)
        for i in range(X.shape[0]):
            output[0][i] = X[i]
        return output

    @staticmethod
    def mul_by_skalar(matrix, skalar):
        return np.asarray([x * skalar for x in matrix])

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def transpose_matrix(X):
        output = np.zeros((X.shape[1], X.shape[0]), dtype=float)
        i, j = (X.shape[0], X.shape[1]) if X.shape[0] < X.shape[1] else (X.shape[1], X.shape[0])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                output[j, i] = X[i, j]
        return output

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def hadamard_matrix(X, Y):
        output = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                output[i, j] = X[i, j] * Y[i, j]
        return output

    @staticmethod
    def hadamard_vector(X, Y):
        output = np.zeros((X.shape[0], 1), dtype=float)
        for i in range(X.shape[0]):
                output[i] = X[i] * Y[i]
        return output

    def validate_model(self, hidden_weights, output_weights, input_values, labels):
        correctly_classified = 0
        errors = 0
        for input, label in zip(input_values, labels):
            output_vector = self.evaluate_model(hidden_weights, output_weights, input)
            predicted_class = self.classify(output_vector)
            if predicted_class == self.classify(label):
                correctly_classified += 1
            else:
                errors += 1
        print "Correctly classified: {}, errors: {}".format(correctly_classified, errors)

    def evaluate_model(self, hidden_weights, output_weights, input_vector):
        hidden_actual_input = self.multiply(hidden_weights, input_vector, True)
        hidden_output_vector = self.sigmoid(hidden_actual_input)
        output_actual_input = self.multiply(self.transpose_matrix(output_weights), hidden_output_vector, False)
        return self.sigmoid(output_actual_input)

    def classify(self, output_vector):
        curr_class = 0
        current_max = 0
        for i in range(output_vector.shape[0]):
            if current_max < output_vector[i]:
                current_max = output_vector[i]
                curr_class = i
        return curr_class

    @staticmethod
    def make_tuples(input, label):
        dataset = list()
        for inp, lab in zip(input, label):
            dataset.append((inp, lab))
        return dataset
