import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel
import math
import argparse


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
    Calculate train and validation perplexities given lists of losses
    """
    train_perplexities = [math.exp(l) for l in train_losses]
    val_perplexities = [math.exp(l) for l in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """

        indices = indices.to(device)
        lengths = lengths.to(device)

        logits = model(indices, lengths)
        logits = logits[:, :-1, :]  
        targets = indices[:, 1:lengths.max()]
        loss = criterion(logits.permute(0, 2, 1), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * indices.size(0)
    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """

        indices = indices.to(device)
        lengths = lengths.to(device)

        logits = model(indices, lengths)
        logits = logits[:, :-1, :]
        targets = indices[:, 1:lengths.max()]
        loss = criterion(logits.permute(0, 2, 1), targets)

        val_loss += loss.item() * indices.size(0)

    val_loss /= len(loader.dataset)
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5,
          save_path: Optional[str] = None, save_every: int = 1):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())

        # checkpoint saving: сохраняем словарь с состоянием модели и оптимизатора
        if save_path is not None and (epoch % save_every == 0):
            ckpt_path = save_path if save_every == 1 else f"{save_path}.epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')


    def _build_dataloaders(data_file: str, sp_prefix: str, vocab_size: int, batch_size: int):
        from dataset import TextDataset
        train_ds = TextDataset(data_file, train=True, sp_model_prefix=sp_prefix, vocab_size=vocab_size)
        val_ds = TextDataset(data_file, train=False, sp_model_prefix=sp_prefix, vocab_size=vocab_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        return train_ds, train_loader, val_ds, val_loader


    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Train RNN language model')
        parser.add_argument('--data', type=str, default='jokes.txt', help='Path to text data file')
        parser.add_argument('--sp_prefix', type=str, default='bpe', help='SentencePiece model prefix')
        parser.add_argument('--vocab_size', type=int, default=2000)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--embed_size', type=int, default=256)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--rnn_type', choices=['RNN', 'LSTM'], default='LSTM')
        parser.add_argument('--save_path', type=str, default='rnn_model_checkpoint.pt',
                            help='Path prefix for saving checkpoints')
        parser.add_argument('--save_every', type=int, default=1,
                            help='Save checkpoint every N epochs (1 = every epoch)')
        parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
        args = parser.parse_args()

        device = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() else torch.device('cuda')

        train_ds, train_loader, val_ds, val_loader = _build_dataloaders(
            args.data, args.sp_prefix, args.vocab_size, args.batch_size
        )

        rnn_cls = torch.nn.LSTM if args.rnn_type == 'LSTM' else torch.nn.RNN
        model = LanguageModel(train_ds, embed_size=args.embed_size, hidden_size=args.hidden_size,
                              rnn_type=rnn_cls).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Запуск обучения с автосохранением чекпойнтов
        train(model, optimizer, scheduler=None, train_loader=train_loader, val_loader=val_loader,
              num_epochs=args.epochs, num_examples=3, save_path=args.save_path, save_every=args.save_every)
