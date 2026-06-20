import os 

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from train import Net, evaluate
from train_hf import get_hf_loaders

FP32_WEIGHTS = "hf_mnist_net.pt"
INT8_WEIGHTS = "hf_mnist_net_int8.pt"

def main():
    torch.backends.quantized.engine = "qnnpack"

    _, test_loader = get_hf_loaders()

    model = Net()
    state_dict = torch.load(FP32_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    fp32_accuracy = evaluate(model, test_loader, device="cpu")

    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    int8_accuracy = evaluate(quantized_model, test_loader, device="cpu")

    torch.save(quantized_model.state_dict(), INT8_WEIGHTS)

    fp32_size_kb = os.path.getsize(FP32_WEIGHTS) / 1024
    int8_size_kb = os.path.getsize(INT8_WEIGHTS) / 1024

    print(f"fp32 accuracy: {fp32_accuracy:.4f}")
    print(f"int8 accuracy: {int8_accuracy:.4f}")
    print(f"fp32 size: {fp32_size_kb:.1f} KB")
    print(f"int8 size: {int8_size_kb:.1f} KB")  

if __name__ ==  "__main__":
    main()
