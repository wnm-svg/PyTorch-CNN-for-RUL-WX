import numpy as np
import matplotlib.pyplot as plt
import time 


def scoring(predictions, max_cycle_t, y_test):
    n_test = 248
    result = []
    ix = 1
    k = 0
    while k < n_test:
        # take the last prediction as the predicted RUL
        result.append((predictions[max_cycle_t[k] + ix - 15 - 1])) 
        # t = max_cycle_t[k] + ix - 15 - 1 : id of last example of every engine 16\51\
        ix = max_cycle_t[k] + ix - 14
        k += 1

    j = 0
    rmse = 0
    score = 0
    while j < n_test:
        h = result[j] - y_test[j]
        rmse = rmse + pow(h, 2)
        if h < 0:
            score += np.exp(-h / 13) - 1
        else:
            score += np.exp(h / 10) - 1
        j += 1

    rmse = np.sqrt(rmse / n_test)
    rmse, score = round(np.asscalar(rmse), 3), round(np.asscalar(score))
    return result, rmse, score


def visualization(y_test, result, root_mse):
    real_time = time.asctime()
    plt.figure(figsize=(25, 10))  # plotting
    plt.axvline(x=248, c='r', linestyle='--')  # size of the training set

    plt.plot(y_test, label='Actual Data')  # actual plot
    plt.plot(result, label='Predicted Data')  # predicted plot
    plt.title('Remaining Useful Life Prediction')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig('./_trials/FD004-RMSE = {}-{}.png'.format(root_mse,real_time))
    plt.show()
