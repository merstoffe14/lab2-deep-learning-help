import matplotlib.pyplot as plt
import torch
import torchvision

from options.classification_options import ClassificationOptions


class MNISTDataset:
    def __init__(self, options: ClassificationOptions):
        self.options = options

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                              transform=transform)
        testset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                             transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=options.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=options.batch_size_test, shuffle=True)

    def show_examples(self):
        """
        Show some examples of the selected dataset.
        """
        examples = enumerate(self.train_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        print(f"The shape of the training data tensor is: {example_data.shape}")

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
