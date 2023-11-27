import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from options.linear_regression_options import LinearRegressionOptions


class HousesDataset(Dataset):
    def __init__(self, options: LinearRegressionOptions, train: bool = True):
        self.options = options
        if train:
            self.dataset_size = options.train_dataset_size
        else:
            self.dataset_size = options.test_dataset_size

        # create data and ensure that it is of shape (batch_size, 2)
        house_sizes = self.create_house_sizes(self.dataset_size, options).unsqueeze(1)
        house_prices = self.create_house_prices_tensor(house_sizes.squeeze(), self.dataset_size, options).unsqueeze(1)
        self.data = torch.cat((house_sizes, house_prices), 1)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :]

    def create_house_prices_tensor(self, house_sizes: torch.Tensor, size: int,
                                   options: LinearRegressionOptions) -> torch.Tensor:
        return (100000 + house_sizes * 5000 + self.options.noise_house_data * torch.randn(size).to(options.device))

    def create_house_sizes(self, size: int, options: LinearRegressionOptions) -> torch.Tensor:
        return (torch.rand(size) * (options.max_house_size - options.min_house_size)
                + options.min_house_size).to(options.device)

    def plot_data(self):
        """
        Show some examples of the selected dataset.
        """

        print(f"The shape of the data tensor is: {self.data.shape}")

        fig = plt.figure()
        plt.scatter(self.data.cpu()[:, 0], self.data.cpu()[:, 1])

        plt.title("Data")
        plt.xlabel("size [m^2]")
        plt.ylabel("Price [€]")

        plt.show()
