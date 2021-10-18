import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = "/storage/users/assafzar/Raz/data/Mitochondria/train"
VAL_DIR = "/storage/users/assafzar/Raz/data/Mitochondria/val"
LEARNING_RATE = 2e-4
PATCH_SIZE = 256
STEP_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 2
CHANNELS_IMG = 1
L1_LAMBDA = 100
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
TRAIN_FRACTION = 0.8
CHECKPOINT_DISC = 'disc.pth.tar'
CHECKPOINT_GEN = 'gen.pth.tar'

both_transform = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value = 255.0,),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

