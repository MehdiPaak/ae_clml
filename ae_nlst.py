import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
import logging
import sys
import os
import pydicom as pyd
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# clearml
# Hyperdatset from allegro
#from allegroai import Task
from clearml import Task, Dataset

#

# Define the autoencoder architecture
class Autoencoder(pl.LightningModule):
    def __init__(self, learning_rate, filter_factor, kernel_size):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.filter_fact = filter_factor
        self.kernel_size = kernel_size


        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32*self.filter_fact, kernel_size=self.kernel_size),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32*self.filter_fact, 64*self.filter_fact, kernel_size=self.kernel_size),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64*self.filter_fact, 128*self.filter_fact, kernel_size=self.kernel_size),
            nn.ReLU(),
            # nn.Conv2d(128*self.filter_fact, 512*self.filter_fact, kernel_size=self.kernel_size),
            # nn.ReLU()

        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512*self.filter_fact, 128*self.filter_fact, kernel_size=self.kernel_size),
            # nn.ReLU(),
            nn.ConvTranspose2d(128*self.filter_fact, 64*self.filter_fact, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(64*self.filter_fact, 32*self.filter_fact, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(32*self.filter_fact, 1, kernel_size=self.kernel_size),
            nn.ReLU()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        #logger.info(f"train_loss: {loss}")

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        #logger.info(f"val_loss: {loss}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class CustomImageDataset(Dataset):
    def __init__(self, img_csv, img_dir, transform=None):
        self.img_files = pd.read_csv(img_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # dicom conversion to pixel array happens here. TODO move to preprocessing step.
        # This was to get it run in the other mlops platform
        img_path = os.path.join(self.img_dir, self.img_files.iloc[idx, 0])
        image_arr = pyd.dcmread(img_path).pixel_array
        image_arr = (image_arr - image_arr.min())/(image_arr.max() - image_arr.min())
        image_arr = image_arr.astype("float32")

        if self.transform:
            image = self.transform(image_arr)
        return image

def main(hparams):
    # print(" CUDA:", torch.cuda.is_available())
    # print(" CUDA devices:", torch.cuda.device_count())

    task = Task.init(
        project_name = "mehdi_test_nlst",
        task_name = "sample_nlst_20k_git_mp"
    )

    # get dataset
    dataset_name = "s3_nlst_20K_1"
    dataset_project = "mehdi_test_nlst/clml_dataset"
    dataset_id = "bb4cd618171248abaa04abf6c56dd17b"

    dataset_path = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        only_completed=False
        ).get_local_copy()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Define the batch size and the number of workers for data loading
    epochs = hparams.epochs
    batch_size = hparams.batch_size
    image_resize = (hparams.image_resize, hparams.image_resize)
    kernel_size = hparams.kernel_size
    learning_rate = hparams.learning_rate
    filter_factor=hparams.filter_factor
    train_size = hparams.train_size
    val_size = hparams.val_size
    log_path = hparams.log_path
    img_csv = os.path.join(dataset_path, "csv", "nlst_20k.csv")#hparams.img_csv
    img_dir = dataset_path #hparams.img_dir
    num_gpu = hparams.numgpu

    logger = TensorBoardLogger(log_path, name="my_model")
    #logger = CSVLogger(log_path, name="ae_nlst")

    num_workers = 2

    # data
    transform_dcm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_resize),
        #transforms.CenterCrop(10),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = CustomImageDataset(img_csv=img_csv, img_dir=img_dir, transform=transform_dcm)
    nlst_train = Subset(dataset=dataset, indices=range(train_size)) #512
    nlst_val = Subset(dataset=dataset, indices=range(train_size, train_size + val_size)) # 512, 640

    train_loader = DataLoader(nlst_train, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(nlst_val, batch_size=batch_size, num_workers=num_workers)

    # Initialize the model
    model = Autoencoder(learning_rate=learning_rate, filter_factor= filter_factor, kernel_size=kernel_size)

    # training
    if device=="cuda":
        trainer = pl.Trainer(devices=num_gpu, accelerator="gpu",
                            num_nodes=1,
                            strategy="ddp",
                            check_val_every_n_epoch=1,
                            max_epochs=epochs,
                            log_every_n_steps=1,
                            logger=logger)
    elif device == "cpu":
            trainer = pl.Trainer(devices="auto", accelerator="cpu",
                            num_nodes=1,
                            check_val_every_n_epoch=1,
                            max_epochs=epochs,
                            log_every_n_steps=1,
                            logger=logger)

    trainer.fit(model, train_loader, val_loader)


# Define the main function
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--filter_factor", type=int, default=1)
    parser.add_argument("--image_resize", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=32)
    parser.add_argument("--val_size", type=int, default=16)
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--img_csv", type=str, default="./data/files_sample.csv")
    parser.add_argument("--img_dir", type=str, default="./data/nlst_sample/")
    parser.add_argument("--numgpu", type=int, default=1)


    args = parser.parse_args()


    main(args)

