import numpy as np
import torch

def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def convert_windows_to_tensor(train_data, dev_data, test_data, window_size, device):
    train_windows = create_windows(train_data, window_size)
    validation_windows = create_windows(dev_data, window_size)
    test_windows = create_windows(test_data, window_size)
    
    train_windows_tensor = torch.tensor(train_windows, dtype=torch.float32).to(device)
    validation_windows_tensor = torch.tensor(validation_windows, dtype=torch.float32).to(device)
    test_windows_tensor = torch.tensor(test_windows, dtype=torch.float32).to(device)
    
    return train_windows_tensor, validation_windows_tensor, test_windows_tensor
