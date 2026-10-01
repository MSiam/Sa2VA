from xtuner.tools.train import main as train
import projects.llava_sam2.hooks.visualization_hook
try:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass
if __name__ == '__main__':
    train()
