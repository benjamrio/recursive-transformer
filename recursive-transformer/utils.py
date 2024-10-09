import torch

def autodetect_device():
    device = 'cpu'
    if torch.cuda.is_available():
        print(f'{torch.cuda.device_count()} cuda devices available')
        device = 'cuda:1'
    return(device)
