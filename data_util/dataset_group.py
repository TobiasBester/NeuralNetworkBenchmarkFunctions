class DatasetGroup:

    def __init__(self, train_dataset, test_dataset, train_labels, test_labels):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_labels = train_labels
        self.test_labels = test_labels
