from MLPModel import MLPModel
from load_data import create_dataset
import numpy as np
numberOfHiddenUnits = 40
learningRate = 0.1
batchSize = 100
epochs = 2


print ('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits)
print ('Learning rate: %d.\n', learningRate)

if __name__ == "__main__":
    trainX, trainY = create_dataset()
    print trainX[0]
    sample_matrix = np.zeros((10, 5), dtype=float)
    sample = MLPModel.mul_by_skalar(sample_matrix, 5)
    print sample.shape
    model = MLPModel(numberOfHiddenUnits, learningRate, batchSize, epochs)

    hiddenWeights, outputWeights, error = model.trainModel(trainX, trainY)

    model.validate_model(hiddenWeights, outputWeights, trainX, trainY)

    # correctlyClassified, classificationErrors = model.validateModel(hiddenWeights, outputWeights, validationX, validationY)

    # print 'Classification errors: %d\n', classificationErrors
    # print 'Correctly classified: %d\n', correctlyClassified
