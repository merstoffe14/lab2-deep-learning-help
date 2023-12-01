class Options:
    def __init__(self):
        # runtime related
        self.random_seed = 1
        self.device = "cpu"

        # model related
        self.save_path = "./models/"
        self.load_path = "./models/"
        # change this to change the model that we want to use.
        # self.model_name = "lin_reg.pth"
        self.model_name = "classification.pth"       