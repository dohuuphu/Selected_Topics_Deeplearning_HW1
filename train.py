import os
import timm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from termcolor import colored
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.utils import *
from utils.dataloader import ImageList, ImageList_test
# from utils.preprocess import val_transform





def main():
    """Main function to train, validate, and test the classification model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Classification")
    # Data parameters
    parser.add_argument(
        "--train_dir", type=str, default="data/train",
        help="Path to the training directory"
    )
    parser.add_argument(
        "--test_dir", type=str, default="data/test",
        help="Path to the test directory"
    )
    parser.add_argument(
        "--val_dir", type=str, default="data/val",
        help="Path to the validation directory"
    )
    # Model parameters
    parser.add_argument(
        "--model", type=str,
        default="timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288", #"timm/resnet101.a1_in1k", #"timm/resnet50.a1_in1k",
        help="Model name from timm"
    )

    # HEAD selection
    parser.add_argument(
        "--head", type=str,
        default="Transformer",
        help="HEAD selection",
        choices=['1_FC', '2_FC', 'Transformer'] 
    )

    # Resize selection
    parser.add_argument(
        "--rs", type=str,
        default="Random_Resized",
        help="Resize selection",
        choices=['Random_Resized', 'Center_Crop', 'Padding'] 
    )

    # Data augmentation
    parser.add_argument(
        "--aug", action="store_true",
        help="Enable data augmentation"
    )

    # Training parameters
    parser.add_argument(
        "--bz", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    # Save directory parameters
    parser.add_argument(
        "--save_dir", type=str, default="new_exp/seresnexta_transformer_resume",
        help="Directory to save checkpoints and logs"
    )

    args = parser.parse_args()

    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(colored(
        f"Loading model: {args.model}",
        color="red", force_color=True
    ))

    # Prepare save directory and logging
    save_dir = f"{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # Initialize logger
    writer = SummaryWriter(log_dir=save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load backbone and HEAD
    if args.head == '1_FC':
        model = timm.create_model(
            args.model, pretrained=True, num_classes=100
        )
    elif args.head =='2_FC':
        model = timm.create_model(
            args.model, pretrained=True
        )
        model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4),
                nn.Linear(2048, 100)
                )
    else:
        model = timm.create_model(
            args.model, pretrained=True
        )
        encoder_layer = TransformerEncoderLayer(d_model=model.fc.in_features//2, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        transformer = TransformerEncoder(encoder_layer, num_layers=1)
        model.fc = nn.Sequential(
                                nn.Linear(model.fc.in_features, model.fc.in_features // 2),  # Bottleneck to reduce dimmesion (fit with 100M #parrams)
                                nn.LeakyReLU(0.1),
                                transformer,  
                                nn.Linear(model.fc.in_features // 2, 100)  # Output layer
                                )

    # Check model size and move to device
    check_model_size(model)
    model.to(device)
    model.train()

    # Load datasets
    print(colored(
        f"Loading datasets from {args.train_dir}, {args.val_dir}, "
        f"{args.test_dir}",
        color="blue", force_color=True
    ))

    input_size = model.default_cfg['input_size'][1]

    train_dataset = ImageList(
        args.train_dir, transform_w=train_transform(resize_size=input_size, rs_type=args.rs, data_augmentation=args.aug)
    )
    val_dataset = ImageList(
        args.val_dir, transform_w=val_transform(resize_size=input_size, rs_type=args.rs)
    )
    test_dataset = ImageList_test(
        args.test_dir, transform=val_transform(resize_size=input_size, rs_type=args.rs)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.bz,
        sampler=sampler_imbalance(train_dataset),
        num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bz,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.bz,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)


    best_valid_acc = 0.0
    max_iters = len(train_loader)

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Training epochs"):
        for step, batch_train in enumerate(train_loader):
            # Move training data to device
            train_w = batch_train["img_w"].to(device)
            train_labels = batch_train["target"].to(device)
            lr = optimizer.param_groups[0]["lr"]

            # Forward pass
            outputs = model(train_w)
            loss = criterion(outputs, train_labels)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training status every 20 steps or at the last iteration
            if step % 20 == 0 or step == max_iters - 1:
                print(
                    "Epoch {} Iters: ({}/{}) \t Loss = {:<10.6f} \t "
                    "learning rate = {:<10.6f}".format(
                        epoch, step, max_iters, loss.item(), lr
                    )
                )
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)

        # Step the learning rate scheduler at the end of the epoch
        scheduler.step()

        # Validation phase
        print(f"Validating at epoch {epoch}")
        val_acc = validate(model, val_loader, device)
        print(f"Validation accuracy: {val_acc * 100:.3f}%")
        writer.add_scalar('Accuracy/val', val_acc, epoch)


        # Save model if is the best val_acc
        if val_acc >= best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val accuracy: {best_valid_acc:.5f}")

            # Inference the model and save predictions
            print("Testing...")
            inference(model, test_loader, device,save_dir)

        # Log best validation accuracy
        log_str = (
            "=====================================\n"
            f"Best Validation Accuracy: {best_valid_acc * 100:.3f}%\n"
            "====================================="
        )
        print(colored(log_str, color="red", force_color=True))


if __name__ == "__main__":
    main()
