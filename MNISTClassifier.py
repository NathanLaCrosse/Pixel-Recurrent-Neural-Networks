import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm

def manual_evaluation(filepath):
    eval_dataset = md.PixelDataset(filepath="Datasets/mnist_test.csv")
    checkpoint = torch.load(f'Models/{filepath}')

    config = checkpoint['config']
    input_size, embedding_size, hidden_size, num_layers, forcing = config.values()

    model = na.MNISTClassifier(input_size, embedding_size,
                               hidden_size, num_layers, True, 10)
    model.load_state_dict(checkpoint['model'])
    im, label = eval_dataset.__getitem__(np.random.randint(0, len(eval_dataset)))

    model = model.eval()
    with torch.no_grad():
        logits = model(im.view(1, 7, 7, 16))

        max_dex = torch.argmax(logits).item()

    im = im.numpy()
    image = im.reshape(7, 7, 4, 4)
    image = image.transpose(0, 2, 1, 3)
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.show()
    print(f'Predicted Label : {max_dex}   True Label : {label}')
    print()


def evaluate(train_network, dataset, filepath):
    if not filepath:
        eval_dataset = dataset
        model = train_network
    else:
        eval_dataset = md.PixelDataset(filepath = "Datasets/mnist_test.csv")
        checkpoint = torch.load(f'Models/{filepath}')

        config = checkpoint['config']
        input_size, embedding_size, hidden_size, num_layers, forcing = config.values()

        model = na.MNISTClassifier(input_size, embedding_size,
                                   hidden_size, num_layers, True, 10)
        model.load_state_dict(checkpoint['model'])

    dat_loader = DataLoader(eval_dataset, batch_size=100, shuffle=True)
    model = model.eval()
    with torch.no_grad():
        correct = 0
        for bat in tqdm(dat_loader):
            im, labels = bat
            logits = model(im.view(100, 7, 7, 16))

            max_dex = torch.argmax(logits, dim = 1)
            correct += (max_dex == labels).sum().item()

        correct = correct / len(eval_dataset)

        print(f"Test Accuracy: {correct * 100}%")
        print()

def train_pixel_rnn(epochs = 1, batch_size = 100, learning_rate = 0.001, input_size = 16,
                    embedding_size = 32, hidden_size = 64, num_layers = 2,
                    omnidirectionality = True, class_count = 10, model_name = 'Omni.pt'):
    pixel_dataset = md.PixelDataset()
    # eval_dataset = md.PixelDataset(filepath = "Datasets/mnist_test.csv")
    model = na.MNISTClassifier(input_size, embedding_size, hidden_size, num_layers,
                               omnidirectionality, class_count)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        dat_loader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=True)
        progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        model = model.train()
        for _, batch in enumerate(progress_bar):
            ims, label = batch

            optimizer.zero_grad()
            output = model(ims)

            loss = loss_fn(output, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})
            model = model.train()
            running_loss += loss.item()
        # evaluate(train_network=model, dataset=eval_dataset,
        #          filepath=None, input_size=input_size, embedding_size=embedding_size,
        #          hidden_size=hidden_size, classification_count=classification_count)

    na.save_checkpoint(input_size, embedding_size, hidden_size, num_layers, None, model, model_name)
