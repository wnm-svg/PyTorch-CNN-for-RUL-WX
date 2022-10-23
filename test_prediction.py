import torch
import numpy as np


def test_prediction(model, test_x):
    # generating predictions for test set
    with torch.no_grad():
        output = model(test_x)


    softmax = torch.exp(output).cpu()
    # print(softmax[0:1])
    prob = list(softmax.numpy())
    # print(prob[0:1])
    # index of max value 
    predictions = np.argmax(prob, axis=1)
    # print(predictions[0:1])

    return predictions
