import torch

FP_PRECISION = torch.float32

def set_precision(precision):
    if precision == "16":
        FP_PRECISION = torch.float16
    if precision == "32":
        FP_PRECISION = torch.float32
    if precision == "64":
        FP_PRECISION = torch.float64