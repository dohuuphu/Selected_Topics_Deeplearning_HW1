import os
import torch
import zipfile
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from termcolor import colored
from collections import Counter
from torch.utils.data import WeightedRandomSampler



def sampler_imbalance(train_dataset):
    # Retrieve labels from the training dataset
    labels = train_dataset.labels

    # Count the frequency of each label using Counter
    counts = Counter(labels)

    # Calculate class weights: if a class does not appear, assign a weight of
    # 0.0, otherwise assign the inverse of its count.
    class_weights = {}
    for c in range(100):
        if counts[c] == 0:
            class_weights[c] = 0.0
        else:
            class_weights[c] = 1.0 / counts[c]

    # Generate a weight for each sample based on its corresponding class weight
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # Create the WeightedRandomSampler using the computed sample weights.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def check_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Print model size in millions using green colored text.
    print(colored(
        f"Total parameters: {total_params/1e6:.2f} M",
        color="green",
        force_color=True
    ))
    # Ensure model size does not exceed 100M parameters.
    assert total_params <= 100e6, (
        "Model size exceeds 100M parameters! Current size: " +
        f"{total_params/1e6:.2f} M"
    )


def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient computation for validation.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            # Move images and labels to the specified device.
            images = batch_data["img_w"].to(device)
            labels = batch_data["target"].to(device)
            outputs = model(images)

            # Obtain predictions by selecting the index with the maximum logit.
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate predictions and labels from all batches.
    labels_tensor = torch.cat(all_labels, dim=0)
    predicts_tensor = torch.cat(all_preds, dim=0)

    # Calculate accuracy.
    accuracy = (
        torch.sum(predicts_tensor == labels_tensor).item() /
        len(labels_tensor)
    )
    model.train()
    return accuracy


def inference(model, dataloader, device, save_dir):
    model.eval()
    predictions = []

    # Disable gradient computation during testing.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Testing"):
            images = batch_data["img"].to(device)
            outputs = model(images)

            # Get the predicted class indices.
            _, preds = torch.max(outputs, 1)

            # Collect predictions with their corresponding image names.
            image_names = batch_data["image_name"]
            for img_name, pred in zip(image_names, preds):
                predictions.append({
                    "image_name": img_name,
                    "pred_label": int(pred)
                })

    # Save predictions to a CSV file.
    df = pd.DataFrame(predictions)
    output_csv = os.path.join(save_dir, "prediction.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # Create a zip archive containing the CSV file.
    zip_filename = (
        f"{save_dir}/prediction.zip"
    )
    with zipfile.ZipFile(
        zip_filename, 'w', zipfile.ZIP_DEFLATED
    ) as zipf:
        zipf.write(output_csv, arcname="prediction.csv")

    print(
        f"Saved zipped predictions to {zip_filename}"
    )
    model.train()


# Transform
def train_transform(resize_size=256, rs_type='Random_Resized', data_augmentation=True):
    transforms_list = [ transforms.ToTensor(),
                        transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                    ]

    if data_augmentation:
        transforms_list.extend( [transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=15),
                                transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))]
                                )
    # Resize type
    if rs_type =='Random_Resized':
        transforms_list.append(transforms.RandomResizedCrop(size=(resize_size, resize_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC))
    elif rs_type == 'Center_Crop':
        transforms_list.extend([transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop((resize_size, resize_size))]
                                )
    else: # Padding
        transforms_list.extend([transforms.Lambda(lambda img: resize_longest_side(img, target_size=resize_size)),  # Resize with longer edge 
                                transforms.Lambda(lambda img: pad_to_square(img)),  # Pad to square image
                                transforms.Resize((resize_size, resize_size))]
                                )
        
    return transforms.Compose(transforms_list)

def val_transform(resize_size=256, rs_type=None):
    transforms_list = [ transforms.ToTensor(),
                        transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                    ]

    # Resize type
    if rs_type =='Random_Resized':
        transforms_list.append(transforms.RandomResizedCrop(size=(resize_size, resize_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC))
    elif rs_type == 'Center_Crop':
        transforms_list.extend([transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop((resize_size, resize_size))]
                                )
    else: # padding
        transforms_list.extend([transforms.Lambda(lambda img: resize_longest_side(img, target_size=resize_size)),  # Resize with longer edge 
                                transforms.Lambda(lambda img: pad_to_square(img)),  # Pad to square image
                                transforms.Resize((resize_size, resize_size))]
                                )
    print(transforms_list)
    return transforms.Compose(transforms_list)

# def val_transform(resize_size=256, crop_size=224):
#     return transforms.Compose([
#         # transforms.Resize((resize_size, resize_size)),
#         # transforms.CenterCrop(crop_size),

#         # crop
#         transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop((resize_size, resize_size)),

#         # padding
#         # transforms.Lambda(lambda img: resize_longest_side(img, target_size=resize_size)),  # Resize theo cạnh dài
#         # transforms.Lambda(lambda img: pad_to_square(img)),  # Pad thành ảnh vuông
#         # transforms.Resize((resize_size, resize_size)),  # Resize về kích thước chuẩn
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[
#                 0.229,
#                 0.224,
#                 0.225,
#             ]
#         )
#     ])


# Resize func
def resize_longest_side(img, target_size):
    """ Resize with longer edge, maintain ratio """
    print('img ', img)
    w, h = img.size
    if w > h:
        new_w, new_h = target_size, int(h * target_size / w)
    else:
        new_w, new_h = int(w * target_size / h), target_size
    return img.resize((new_w, new_h), Image.BICUBIC)

def pad_to_square(img):
    """ Padding to square shape """
    w, h = img.size
    max_side = max(w, h)
    padding_left = (max_side - w) // 2
    padding_right = max_side - w - padding_left
    padding_top = (max_side - h) // 2
    padding_bottom = max_side - h - padding_top

    return transforms.functional.pad(img, (padding_left, padding_top, padding_right, padding_bottom), fill=0)



# SE block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        se = self.global_avg_pool(x).view(b, c)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1, 1)
        return x * se

class SEResNeXtBottleneck(nn.Module):
    def __init__(self, original_block):
        super(SEResNeXtBottleneck, self).__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.act1 
        self.downsample = original_block.downsample
        self.se = SEBlock(original_block.conv3.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def replace_bottlenecks_with_se(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):  # main block of ResNet-101
            for i, sub_module in enumerate(module):
                if isinstance(sub_module, nn.Module) and hasattr(sub_module, 'conv3'):  # Bottleneck block
                    module[i] = SEResNeXtBottleneck(sub_module)
