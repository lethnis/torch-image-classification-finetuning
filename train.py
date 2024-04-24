from torch.utils.data import random_split, DataLoader
from torchinfo import summary

from torchvision import datasets, transforms

import lightning as L

from utils import get_lit_efficientnet, get_lit_shufflenet_0_5, get_lit_shufflenet_1_0


def main():

    # prepare images
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # create a dataset from folder
    dataset = datasets.ImageFolder("../datasets/animals10", transform=transform)

    # split dataset
    train_ds, test_ds = random_split(dataset, lengths=[0.9, 0.1])

    # create dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # create model as lightning module
    model = get_lit_shufflenet_1_0()

    # print summary to view model
    summary(model, input_size=(16, 3, 224, 224), col_names=["output_size", "num_params", "trainable"])

    # create trainer
    trainer = L.Trainer(max_epochs=5, enable_model_summary=False)
    # start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
