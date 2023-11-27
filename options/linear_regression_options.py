from options.options import Options


class LinearRegressionOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 10
        self.batch_size_train = 20
        self.train_dataset_size = 100
        self.test_dataset_size = 20
        self.min_house_size = 30
        self.max_house_size = 70
        self.noise_house_data = 50000

        # hyperparameters
        self.lr = 1
        self.num_epochs = 10
