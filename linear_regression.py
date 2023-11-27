import torch
from torch.utils.data import DataLoader

from utilities import utils
from datasets.houses_dataset import HousesDataset
from models.models import LinearRegression
from options.linear_regression_options import LinearRegressionOptions

if __name__ == "__main__":
    options = LinearRegressionOptions()
    utils.init_pytorch(options)

    # create and visualize datasets
    train_dataset = HousesDataset(options, train=True)
    test_dataset = HousesDataset(options, train=False)
    train_dataset.plot_data()
    test_dataset.plot_data()

    # create dataloaders for easy access
    train_dataloader = DataLoader(train_dataset, options.batch_size_train)
    test_dataloader = DataLoader(test_dataset, options.batch_size_test)

    """START TODO: fill in the missing parts as mentioned by the comments."""
    # create a LinearRegression instance named model
    model = LinearRegression()
    model.to(options.device)
    # define the opimizer
    # (visit https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim#module-torch.optim for more info)\
    learning_rate = 0.001
    # met een learning rate van 0.01 divergeerde de error zo hard dat het naar inf... en NaN ging...
    # dan geprobeerd met een lr van 0.0001 en dan was de uiteindelijke avg error 2738.33

    # "We expect you to perform some hyperparameter tuning; what happens if you increase/decrease the learning rate, the size of the training/test data, the batch size, the noise on the data and the number of epochs?"
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # train the model
    utils.train_lin_model(model, optimizer, train_dataloader, options)

    """END TODO"""

    # test the model
    print("Testing the model...\n")

    print("On the train set:")
    utils.test_lin_reg_model(model, train_dataloader)
    utils.test_lin_reg_plot(model, train_dataloader, options)

    print("On the test set:")
    utils.test_lin_reg_model(model, test_dataloader)
    utils.test_lin_reg_plot(model, test_dataloader, options)
    utils.print_lin_reg(model, options)

    # save the model
    utils.save(model, options)
