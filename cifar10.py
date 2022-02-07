import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flows.flows import Flow
from utils.progress import ProgressBar

# train_dataset = CIFAR10(train=True, download=False, root='/prj/neo-nas/databatch/Cifar10/cifar-10-batches-py')
# test_dataset = CIFAR10(train=False, download=False, root='/prj/neo-nas/databatch/Cifar10/cifar-10-batches-py')
# print(len(train_dataset), len(test_dataset))

flow_dict = {
    'made'
}


class ConditonalNN(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Cifar10Trainer(pl.LightningModule):
    def __init__(self, n_blocks, train_batch_size, val_batch_size, learning_rate=1e-3,
                 weight_decay=1e-4, noise_growth_rate=0):
        super().__init__()
        self.n_blocks = n_blocks
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.n_classes = 10
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.img_shape = 32

        self.cond_nn = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        self.flows = Flow(
            n_classes=self.n_classes,
            n_blocks=self.n_blocks,
            n_hidden=100,
            cond_nn_factory=ConditonalNN,
            noise_growth_rate=noise_growth_rate,
            reject_sampling=False
        )

        self.dataset_root = '/prj/neo-nas/databatch/Cifar10/cifar-10-batches-py'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def forward(self, x, y):
        log_prob = self.flows.log_probs(y, cond_inputs=x)
        loss = -log_prob.sum()
        return loss

    def reverse(self, x, y):
        y_, _ = self.flows.sample_with_log_prob(n_samples=y.shape[0], cond_inputs=x)
        return y_

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = F.one_hot(batch[1], num_classes=self.n_classes).float()
        train_loss = self(inputs, labels)
        self.log('train loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = F.one_hot(batch[1], num_classes=self.n_classes).float()
        preds = self.reverse(inputs, labels)
        val_acc = torch.mean(torch.Tensor.float(labels.argmax(dim=1) == preds.argmax(dim=1))).item()
        val_loss = self(inputs, labels)
        self.log('top-1 val acc', val_acc, prog_bar=True)
        self.log('val loss', val_loss, prog_bar=True)
        return {'top-1 val acc': val_acc}

    def train_dataloader(self):
        dataset = CIFAR10(train=True, download=False, root=self.dataset_root, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_batch_size, num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataset = CIFAR10(train=False, download=False, root=self.dataset_root, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.val_batch_size, num_workers=4)
        return dataloader

    def test_dataloader(self):
        dataset = CIFAR10(train=False, download=False, root=self.dataset_root, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
        return dataloader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def train_cifar10(n_epochs, n_blocks, train_batch_size, val_batch_size, learning_rate, weight_decay, noise_growth_rate):
    bar = ProgressBar()
    ncp_cf = Cifar10Trainer(
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        n_blocks=n_blocks,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        noise_growth_rate=noise_growth_rate
    )
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        # limit_train_batches=10,
        # limit_val_batches=5,
        # limit_test_batches=1,
        gpus=1,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        callbacks=[bar]
    )
    trainer.fit(model=ncp_cf)
    # trainer.test(ncp_cf)
