from dataset_prepare import *
from load_data import *
from model import *
from test_prediction import *


nf = 21
# nf = 15

model = CNN1(nf)
model.load_state_dict(torch.load('./model/model_FD004.pth'))
x=torch.randn((1,1,15,21))
# x=torch.randn((1,1,15,15))

export_onnx_file = "./model/model_FD004.onnx"
torch.onnx.export(
                model, 
                x, 
                export_onnx_file, 
                export_params=True
                )

print("model_FD004.onnx saved")