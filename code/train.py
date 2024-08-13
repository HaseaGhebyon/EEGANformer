from tqdm import tqdm
import numpy as np

from pathlib import Path
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import torcheeg
from torcheeg import model_selection
from torcheeg.datasets import NumpyDataset

from model import build_eegformer, build_discriminator
from config import latest_weights_file_path, get_config, get_database_name, get_weights_file_path

def get_eeg_dataset(config):
    print("Retrieving dataset from database ...\n")
    X = np.random.randn(100, 32, 128)
    y = {
        'valence': np.random.randint(10, size=100),
        'arousal': np.random.randint(10, size=100)
    }
    dataset = NumpyDataset(X=X,
                       y=y,
                       io_path=config["eeg_datasource"],
                       online_transform=torcheeg.transforms.ToTensor(),
                       label_transform=torcheeg.transforms.Compose([
                           torcheeg.transforms.Select('label')
                       ]),
                       num_worker=1,
                       num_samples_per_worker=50)

    train_dataset, test_dataset = model_selection.train_test_split(dataset=dataset, test_size=0.2, random_state=7 )
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    return train_dataloader, test_dataloader

def load_random_image(dataset, label):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for image, lbl in loader:
        if lbl == label:
            return image.squeeze(dim=0)
    print(f"There is no image with label {label}") 
    

def get_image(config, label_seq:list):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    image_dataset = ImageFolder(root=config['image_datasource'], transform=transform)
    result = []
    for seq in label_seq:
        tensor = load_random_image(image_dataset, seq)
        result.append(tensor)
    return torch.stack(result)

def save_predict_image(config, image_tensor, epoch, size=(1,28,28), nrow=4, show=True):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"{config['target_generation']}_{epoch}.png")

def train_model(config, device):
    print("\nConfigur Training process...")
    Path(f"{config['eeg_datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    global_step = 0
    initial_epoch = 0
    preload = config["preload"]
    train_dataloader, test_dataloader = get_eeg_dataset(config)
    writer = SummaryWriter(config['eeg_datasource'] + "/" + config['experiment_name'])

    # ======================== INITIALIZE MODEL ======================== #
    # GENERATOR
    generator = build_eegformer(
        channel_size = len(config["selected_channel"]),
        seq_len = config["seq_len"],
        num_cls = config["num_cls"],
        N = config["transformer_size"]
    ).to(device)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=config['learning_rate'])
    model_generator_filename = latest_weights_file_path(config, "generator") if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_generator_filename:
        print(f'Preloading model Generator : {model_generator_filename}')
        state = torch.load(model_generator_filename)
        discriminator.load_state_dict(state['state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model Generator to preload, starting from scratch")

    # DISCRIMINATOR
    discriminator = build_discriminator()
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'])
    model_discriminator_filename = latest_weights_file_path(config, "discriminator") if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_discriminator_filename:
        print(f'Preloading model Discriminator : {model_discriminator_filename}')
        state = torch.load(model_discriminator_filename)
        discriminator.load_state_dict(state['disc_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model Discriminator to preload, starting from scratch")
    # ======================== INITIALIZE MODEL ======================== #

    criterion = nn.BCEWithLogitsLoss().to(device)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(initial_epoch, config['num_epoch']):
        torch.cuda.empty_cache()

        generator.train()
        discriminator.train()
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", position=0, leave=True)
        for batch in train_dataloader:
            label = batch[1].to(device)
            inputs = batch[0].to(device)
            
            # TRAIN THE DISCRIMINATOR
            real = get_image(config, label)
            real = torch.Tensor(real).to(device)
            
            disc_opt.zero_grad()
            
            fake            = generator(inputs)
            disc_fake_pred  = discriminator(fake.detach())
            disc_real_pred  = discriminator(real)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_fake_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step() 
            writer.add_scalar('Train Disc Loss', disc_loss.item(), global_step) # LOGGING USING TENSOR BOARD
            writer.flush()
            
            # TRAIN THE GENERATOR
            gen_opt.zero_grad()
            disc_fake_pred = discriminator(fake)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()
            writer.add_scalar('Train Gen Loss', gen_loss.item(), global_step) # LOGGING USING TENSOR BOARD
            writer.flush()
            
            
            # DISPLAY
            batch_iterator.set_postfix({
                "Disc. loss": f"{disc_loss.item():6.3f}",
                "Gen. loss" : f"{gen_loss.item():6.3f}"
            })

            global_step +=1

            # SAVE DISPLAY
        
        if epoch % config["saving_epoch_step"] == 0:
            model_generator_filename = get_weights_file_path(config, f"{epoch:02d}", type="generator")
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': gen_opt.state_dict(),
                'global_step': global_step
            }, model_generator_filename)

            model_discriminator_filename = get_weights_file_path(config, f"{epoch:02d}", type="discriminator")
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': disc_opt.state_dict(),
                'global_step': global_step
            }, model_discriminator_filename)

