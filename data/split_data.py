def split_data(scaled_data, train_ratio=0.7, dev_ratio=0.15):
    train_size = int(train_ratio * len(scaled_data))
    dev_size = int(dev_ratio * len(scaled_data))
    test_size = len(scaled_data) - train_size - dev_size

    train_data = scaled_data[:train_size]
    dev_data = scaled_data[train_size:train_size + dev_size]
    test_data = scaled_data[train_size + dev_size:]

    return train_data, dev_data, test_data
