# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim
import onnx

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./checkpoint/snapshot/checkpoint.pth.tar")
parser.add_argument('--onnx_model', default="./output/pfld.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])

# 💡 修正 1：切換至推論模式 (解決黃字警告)
pfld_backbone.eval()
print("PFLD bachbone:", pfld_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["input_1"]
output_names = ["output_1"]

# 💡 順手指定 opset=11 對 Edge 編譯器更友善
torch.onnx.export(pfld_backbone,
                  dummy_input,
                  args.onnx_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=11) 

print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
# 💡 修正 2：用 model_opt, check 接收 Tuple 回傳值
model_opt, check = onnxsim.simplify(args.onnx_model)

if check:
    onnx.save(model_opt, args.onnx_model_sim)
    print("onnx model simplify Ok!")
else:
    print("Warning: ONNX simplification failed validation.")
    # 就算 simplify 失敗，我們還是保存未簡化的版本供後續使用
    onnx.save(model, args.onnx_model_sim)