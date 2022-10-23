from torch import nn, optim
from torch.autograd import Variable

import evaluation
from dataset_prepare import *
from load_data import *
from model import *
from test_prediction import *
import numpy 

train_raw, test_raw, max_cycle, max_cycle_t, y_test = load_data_FD001()
X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t = get_info(train_raw, test_raw)
test_x = test_prepare(Xt_ss, idx_t, nf, ns_t)
model = CNN1(15)   
model.load_state_dict(torch.load('model/model_FD001.pth'))
predictions = test_prediction(model, test_x)
print(len(predictions))