import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.mnist_dataset import MNISTDataset
from losses.losses import mse
from models.models import LinearRegression, Classifier
from options.classification_options import ClassificationOptions
from options.linear_regression_options import LinearRegressionOptions
from options.options import Options


def plot_tensor(to_plot: torch.Tensor, title: str):
    gray_image_tensor = to_plot.view([1, -1, 1])
    numpy_im = gray_image_tensor.cpu().numpy()
    plt.imshow(numpy_im, cmap=plt.get_cmap("GnBu"), interpolation="none", vmin=0,
               vmax=1)
    plt.title(title)
    plt.show()


def plot_rgb_tensor(to_plot: torch.Tensor, title: str):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(transforms.ToPILImage()(to_plot), interpolation="None")
    plt.show()


def train_lin_model(model: LinearRegression, optimizer: torch.optim.Optimizer,
                    train_data: DataLoader, options: LinearRegressionOptions):
    for epoch in range(options.num_epochs):
        for data in train_data:
            # data that gives a size and a price, we run a size trough the model and want a price.
            size, price = data[:, 0].unsqueeze(1).to(options.device), data[:, 1].unsqueeze(1).to(options.device)
            """START TODO: implement some missing parts. look at the comments to see what needs to be done."""
            # Forward the size data through the model
            output = model.forward(size)
            # calculate the loss, use your self created mse loss
            loss = mse(output, price)
            # As mentioned before, the grads always needs to be zeroed before backprop (use your optimizer to do this)
            optimizer.zero_grad()
            # propagate the loss backward
            loss.backward()
            # use your optimizer to perform an update step
            optimizer.step()

            """END TODO"""
        print(f'epoch [{epoch + 1}/{options.num_epochs}]: ', end="")
        test_lin_reg_model(model, train_data)


def test_lin_reg_model(model: LinearRegression, test_data: DataLoader):
    with torch.no_grad():
        loss = 0
        for data in test_data:
            size, price = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)
            estimated_price = model(size)
            loss = loss + torch.sqrt(mse(estimated_price, price))
        print(f'Avg error/example: € {loss / len(test_data.dataset) :.2f}\n')


def print_lin_reg(model: LinearRegression, options: LinearRegressionOptions):
    print(f"Actual function: f(x) = 5000 * x + 100 000 + {options.noise_house_data} * N(0, 1).")
    print(
        f"Estimated by linear regression: h(x) = {model.linear_layer.weight.data[0].item()} * x + {model.linear_layer.bias.data[0].item()}")


def test_lin_reg_plot(model: LinearRegression, test_data: DataLoader, options: LinearRegressionOptions):
    """
    Show some examples of the selected dataset.
    """
    fig = plt.figure()

    # plot real and estimated data points
    with torch.no_grad():
        for data in test_data:
            plt.scatter(data.cpu()[:, 0], data.cpu()[:, 1], c="g")
            size, price = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)
            estimated_price = model(size)
            plt.scatter(data.cpu()[:, 0], estimated_price.cpu(), c="r")

        # plot line
        x = torch.linspace(options.min_house_size, options.max_house_size, 50000, device="cpu")
        plt.plot(x.numpy(), 5000 * x + 100000, "g")
        plt.plot(x.numpy(), model(x.unsqueeze(1).to(options.device)).cpu().numpy(), "r")

    plt.title("Data")
    plt.xlabel("size [m^2]")
    plt.ylabel("Price [€]")
    plt.legend(["Unknown function f(x)", "Linear regression line", "Real data samples", "Estimated data samples"])
    plt.plot()

    plt.show()


def train_classification_model(model: Classifier, optimizer: torch.optim.Optimizer,
                               dataset: MNISTDataset, options: ClassificationOptions):
    """START TODO: select an appropriate criterion (loss function)"""
    # volgens deze website: https://neptune.ai/blog/pytorch-loss-functions
    # The Negative Log-Likelihood Loss function (NLL) is applied only on models with the softmax function as an output activation layer.
    # Multi-class classification problems
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    #NLLLoss had percies problemen met mutli target? probeer nog eens later en zie error, NLLLoss is enkel voor 
    """END TODO"""
    for epoch in range(options.num_epochs):
        running_loss = 0
        ct = 0
        for x, y in dataset.train_loader:
            ct += 1
            # print(ct)
            y = y.to(options.device)
            """START TODO: fill in the gaps as mentioned by the comments"""
            # forward the data x through the model.
            # Note: x does not have the correct shape,
            # it should become (batch_size, -1), where the size -1 is inferred from other dimensions
            # (see TORCH.TENSOR.VIEW on the PyTorch documentation site)
            # print(f"x before view: {x.size()}")
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            output = model.forward(x)
            # calculate the loss, use your previously defined criterion
            # print(f"x after view: {x.size()}")
            # print(f"output: {output.size()}")
            # print(f"y: {y.size()}")
            # print(y)
            #first I had num_classes at -1, so it would be inferred. But this failed at iteration 56, In this batch, none of the outcomes were '9'. So the size of y after onehot became [64,9], 
            # which would make the lossfunction crash
            y = nn.functional.one_hot(y,num_classes=10)
            # print(f"y one hot: {y.size()}")
            # print(y)
            # print(output)
            # deze x en y komen niet overeen van grootte en soort, dus eigenlijk kan je hier geen loss van berekenen, daarom geraak ik maar aan een accuracy van 83 procent.
            # Dus de orginele x is [64, 1, 28, 28]. Dit zijn dus 64 (batch size) zwart wit images van 28*28, hier maken we met .view
            # dit zetten we om met view naar iets van 2 dimensies, de batch size en een inferred dimentie (-1). De inferred dimensie is hier 28*28. Dus nu hebben we een input
            # tensor die 64 1d vectors zijn.
            # de output is per 1d 784 vector een 1d vector van 10 lang (64 keer want batch size). Deze 10 end values zullen door de softmax de kansverdeling zijn van de cijfers 0-9.
            # de gewenste output (y) is een 1d tensor van 64 lang, die op elke positie het juiste cijfer bevat, dus nu moeten we met oneHot, y op output laten lijken (dus [64,10])

            # ik kreeg een type error dus dit zorgt er voor dat ze dezelfde type hebben.
            y = y.float()
            loss = criterion(output,y)
            # zero out all gradients
            optimizer.zero_grad()
            # propagate the loss backward
            loss.backward()
            # use your optimizer to perform an update step
            optimizer.step()
            
            """END TODO"""
            # this is from the regression one, just to compare.
            #  # Forward the size data through the model
            # output = model.forward(size)
            # # calculate the loss, use your self created mse loss
            # loss = mse(output, price)
            # # As mentioned before, the grads always needs to be zeroed before backprop (use your optimizer to do this)
            # optimizer.zero_grad()
            # # propagate the loss backward
            # loss.backward()
            # # use your optimizer to perform an update step
            # optimizer.step()

            """end fill"""
            running_loss += loss.item()
        print(f'epoch [{epoch + 1}/{options.num_epochs}]: ', end="")
        print(f"Running loss = {running_loss / len(dataset.train_loader)}")
        test_classification_model(model, dataset, options)


def test_classification_model(model: Classifier, dataset: MNISTDataset, options: ClassificationOptions):
    with torch.no_grad():
        tot = 0
        correct = 0

        for x, y in dataset.test_loader:
            output = model(x.view(x.shape[0], -1).to(options.device))

            # choose the number with the highest probability as prediction
            _, predicted = torch.max(output, dim=1)
            tot += y.size(0)
            correct += (predicted == y.to(predicted.device)).sum().item()
        print(f'Accuracy: {100 * correct / tot :.2f}%')


def classify_images(model: Classifier, dataset: MNISTDataset, options: ClassificationOptions):
    with torch.no_grad():
        examples = enumerate(dataset.test_loader)
        _, (x, y) = next(examples)

        output = model(x.view(x.shape[0], -1).to(options.device))

        # choose the number with the highest probability as prediction
        predicted = torch.argmax(output, dim=1)

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(x[i][0], cmap='gray', interpolation='none')
            plt.title(f"y: {y[i]}, estimation: {predicted[i].item()}")
            plt.xticks([])
            plt.yticks([])
        plt.show()


def not_implemented() -> str:
    return "NOT IMPLEMENTED"


def save(model: nn.Module, options: Options):
    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)
    torch.save(model.state_dict(), options.save_path + options.model_name)


def load(model: nn.Module, options: Options):
    try:
        model.load_state_dict(torch.load(options.load_path + options.model_name))
        model.eval()
    except IOError:
        print("Could not load module!!")


def init_pytorch(options: Options):
    # set all random seeds for reproducibility
    torch.manual_seed(options.random_seed)
    torch.cuda.manual_seed(options.random_seed)
    random.seed(options.random_seed)
    # set device
    if options.device == "cuda" and torch.cuda.is_available():
        options.device = torch.device("cuda:0")
    else:
        options.device = torch.device("cpu")
