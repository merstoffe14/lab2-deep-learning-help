from datasets.mnist_dataset import MNISTDataset
from models.models import Classifier
from options.classification_options import ClassificationOptions
from utilities import utils
from utilities.utils import init_pytorch, test_classification_model, classify_images

if __name__ == "__main__":
    options = ClassificationOptions()
    init_pytorch(options)

    # create and visualize the MNIST dataset
    dataset = MNISTDataset(options)
    dataset.show_examples()

    # create a Classifier instance named model
    model = Classifier(options)
    model.to(options.device)

    # load the model
    utils.load(model, options)

    # Test the model
    print("The Accuracy of the model is: ")
    test_classification_model(model, dataset, options)
    classify_images(model, dataset, options)
