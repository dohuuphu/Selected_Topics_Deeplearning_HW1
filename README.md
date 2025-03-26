# Model Training and Experiment Configuration

This README provides details on the various configurable parameters for HW1 training a deep learning model.

Environment Setup

Python version: 3.8.17

PyTorch version: 2.0.1
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```




## Arguments

### 1. **Data Parameters**
These parameters specify the directories where your training, validation, and test datasets are stored.

- `--train_dir` (default: `data/train`):
  - **Description**: Path to the training data directory.
  
- `--test_dir` (default: `data/test`):
  - **Description**: Path to the test data directory.
  
- `--val_dir` (default: `data/val`):
  - **Description**: Path to the validation data directory.

### 2. **Model Parameters**
These parameters allow you to select the model to use for training.

- `--model` (default: `timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288`):
  - **Description**: Specify the model to use from the `timm` library. For example:
    - `timm/resnet101.a1_in1k`
    - `timm/resnet50.a1_in1k`
    - `timm/resnet18.a1_in1k`
  - **Note**: This option selects the pre-trained model.

### 3. **Head Selection**
This option determines which type of HEAD to use.

- `--head` (default: `Transformer`):
  - **Description**: Specifies the type of head to use in the model.
  - **Choices**: `1_FC`, `2_FC`, `Transformer`

### 4. **Resize Selection**
This parameter defines how the images will be resized before being input into the model.

- `--rs` (default: `Random_Resized`):
  - **Description**: Selects the resizing method for the input images.
  - **Choices**: `Random_Resized`, `Center_Crop`, `Padding`

### 5. **Data Augmentation**
This option enables or disables data augmentation techniques to improve model generalization.

- `--aug` (flag):
  - **Description**: Enable data augmentation during training.
  - **Action**: `store_true` (use the flag to turn it on).

### 6. **Training Parameters**
These parameters are essential for controlling the training process.

- `--bz` (default: `32`):
  - **Description**: Set the batch size for training.
  
- `--lr` (default: `0.0001`):
  - **Description**: Learning rate for training.
  
- `--epochs` (default: `50`):
  - **Description**: The number of training epochs to run.

### 7. **Save Directory Parameters**
Specify the directory where checkpoints and logs will be saved during training.

- `--save_dir` (default: `new_exp/seresnexta_transformer_resume`):
  - **Description**: Directory to save model checkpoints and logs.

---

## Example Usage

To train a model using the provided parameters, you can execute `run.bash` or the script as follows:

```bash
python train.py \
    --train_dir data/train \
    --test_dir data/test \
    --val_dir data/val \
    --model timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288 \
    --head Transformer \
    --rs Random_Resized \
    --aug \
    --bz 32 \
    --lr 0.0001 \
    --epochs 50 \
    --save_dir experimenet_result

```

To visualize: 
```bash
tensorboard --logdir experimenet_result