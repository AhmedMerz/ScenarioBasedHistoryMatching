import os
# Set the working directory. This is crucial for relative paths (like DATA_DIR) to work correctly.
# It's generally better to use absolute paths or construct paths relative to the script's location,
# but for this specific request, we'll keep the user's specified working directory.
os.chdir('D:/Scenario2') # Keep your working directory

import torch
import torch.nn as nn # Neural network modules (layers, activation functions, etc.)
import torch.optim as optim # Optimization algorithms (Adam, SGD, etc.)
from torch.utils.data import Dataset, DataLoader # For creating custom datasets and loading data in batches
import torchvision.transforms as transforms # Common image transformations
import torchvision.utils as vutils # Utilities for vision tasks, e.g., saving image grids
import torch.autograd as autograd # For automatic differentiation, used in gradient penalty
from torch.nn.utils import spectral_norm # For applying spectral normalization to layers
import numpy as np # For numerical operations, especially with arrays
import matplotlib.pyplot as plt # For plotting graphs and images
import glob # For finding files matching a pattern (not directly used here but often useful)
import re # For regular expressions (not directly used here but often useful for filename parsing)
from sklearn.preprocessing import MinMaxScaler # For scaling numerical features to a specific range (e.g., [0, 1])
import math # For mathematical functions (e.g., math.radians, math.sin, math.cos)
import traceback # For printing detailed error messages

# --- Configuration ---
# This section defines all the hyperparameters and settings for the GAN.

# Data and File Settings
DATA_DIR = './data/'  # Directory where the .npy data files are stored (relative to os.chdir)
FRAC_VALUES = [30, 40, 50, 60, 70] # List of 'frac' parameter values used in filenames and for label encoding.
WIDTH_VALUES = [15, 20, 25, 30]    # List of 'width' parameter values.
ORIENTATION_VALUES = [0, 30, 60, 90] # List of 'orientation' parameter values (in degrees).
# Filename pattern for data files. Uses f-string like placeholders.
# e.g., "30_frac_15_width_0_orient.npy"
FILENAME_PATTERN = "{frac}_frac_{width}_width_{orient}_orient.npy"

# Image Properties
IMG_HEIGHT = 128      # Target height of the images.
IMG_WIDTH = 128       # Target width of the images.
IMG_CHANNELS = 1      # Number of channels in the images (1 for grayscale, 3 for RGB).

# GAN Model Hyperparameters
Z_DIM = 100           # Dimensionality of the latent noise vector (input to Generator).
LABEL_DIM = 4         # Dimensionality of the encoded label vector (frac_norm, width_norm, sin_orient, cos_orient).
G_FEATURES = 64       # Base number of feature maps in the Generator. Layers will use multiples of this.
D_FEATURES = 64       # Base number of feature maps in the Critic/Discriminator.
EMBED_SIZE = 32       # Dimensionality of the embedded labels before concatenation/projection.
                      # Reduced from 128 as it might be too large for a 4D label, potentially leading to overfitting
                      # or dominance of label information. 32 or 64 is a more common choice.

# Training Hyperparameters
NUM_EPOCHS = 10000    # Total number of training epochs.
BATCH_SIZE = 16       # Number of samples per batch. May need reduction if Self-Attention causes Out-Of-Memory (OOM).
LR = 0.0001           # Learning rate for Adam optimizers. Increased from 0.00005 as WGANs can sometimes benefit from slightly higher LRs.
BETA1 = 0.0           # Adam optimizer's beta1 hyperparameter. 0.0 is common in WGAN-GP.
BETA2 = 0.9           # Adam optimizer's beta2 hyperparameter. 0.9 is common in WGAN-GP.
CRITIC_ITERATIONS = 5 # Number of Critic updates per Generator update. Crucial for WGAN stability.
LAMBDA_GP = 10        # Gradient penalty coefficient for WGAN-GP.

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, else CPU.

# Output Directories
OUTPUT_DIR = './cgan_wgan_gp_sagan_final' # Main directory to save outputs.
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')    # Directory to save generated image samples.
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')  # Directory to save trained model checkpoints.
os.makedirs(IMG_DIR, exist_ok=True)   # Create image directory if it doesn't exist.
os.makedirs(MODEL_DIR, exist_ok=True) # Create model directory if it doesn't exist.

# --- Label Handling ---
# This section defines how the conditional labels (frac, width, orientation) are processed.

# Initialize MinMaxScaler for 'frac' and 'width' parameters.
# MinMaxScaler scales features to a given range, typically [0, 1].
# This helps in stabilizing training and ensuring that all parts of the label have a similar numerical range.
frac_scaler = MinMaxScaler()
width_scaler = MinMaxScaler()

# Fit the scalers with the defined parameter values.
# .reshape(-1, 1) is required because MinMaxScaler expects 2D input.
frac_scaler.fit(np.array(FRAC_VALUES).reshape(-1, 1))
width_scaler.fit(np.array(WIDTH_VALUES).reshape(-1, 1))

def encode_labels(frac, width, orientation_deg):
    """
    Encodes the raw frac, width, and orientation values into a normalized tensor.
    - Frac and Width are scaled to [0, 1].
    - Orientation (degrees) is converted to (sin, cos) to handle periodicity
      (e.g., 0 degrees is close to 360 degrees) and provide a continuous representation.
    """
    # Scale frac and width to [0, 1] using the pre-fitted scalers.
    frac_norm = frac_scaler.transform(np.array([[frac]]))[0, 0]
    width_norm = width_scaler.transform(np.array([[width]]))[0, 0]

    # Convert orientation from degrees to radians.
    orientation_rad = math.radians(orientation_deg)
    # Represent orientation using sine and cosine components.
    # This is a common way to represent cyclical features for neural networks.
    sin_orient = math.sin(orientation_rad)
    cos_orient = math.cos(orientation_rad)

    # Return as a PyTorch tensor.
    return torch.tensor([frac_norm, width_norm, sin_orient, cos_orient], dtype=torch.float32)

def decode_labels(label_tensor):
    """
    Decodes a normalized label tensor back to its original frac, width, and orientation_deg values.
    This is useful for interpreting the labels of generated images.
    """
    label_np = label_tensor.cpu().numpy() # Move tensor to CPU and convert to NumPy array.
    # Ensure input is 2D (N, 4) for batch processing, even if a single label (4,) is passed.
    if label_np.ndim == 1: label_np = label_np.reshape(1, 4)
    if label_np.ndim != 2 or label_np.shape[1] != 4:
        raise ValueError(f"Input tensor must have shape (N, 4) or (4,) for decoding. Got {label_np.shape}")

    # Inverse transform frac and width from [0, 1] back to original scale.
    frac = frac_scaler.inverse_transform(label_np[:, 0].reshape(-1, 1)).flatten()
    width = width_scaler.inverse_transform(label_np[:, 1].reshape(-1, 1)).flatten()

    # Extract sin and cos components of orientation.
    sin_orient, cos_orient = label_np[:, 2], label_np[:, 3]
    # Convert (sin, cos) back to radians using arctan2.
    # arctan2 is preferred over arctan as it correctly handles all quadrants and the signs of sin/cos.
    orientation_rad = np.arctan2(sin_orient, cos_orient)
    # Convert radians back to degrees.
    orientation_deg = np.degrees(orientation_rad)

    # Handle potential NaN from arctan2 if sin and cos are both zero (unlikely with normalized labels but good practice).
    orientation_deg = np.nan_to_num(orientation_deg)

    # Return single values if a single label was input, otherwise return arrays.
    if label_tensor.ndim == 1:
        return frac.item(), width.item(), orientation_deg.item()
    else:
        return frac, width, orientation_deg

# --- Custom Dataset ---
class FracWidthOrientImageDataset(Dataset):
    """
    Custom PyTorch Dataset to load images and their corresponding frac, width, orientation labels.
    It scans the DATA_DIR for .npy files matching the FILENAME_PATTERN.
    """
    def __init__(self, data_dir, frac_values, width_values, orientation_values, filename_pattern, img_height, img_width):
        self.data_dir = data_dir
        self.frac_values = frac_values
        self.width_values = width_values
        self.orientation_values = orientation_values
        self.filename_pattern = filename_pattern
        self.img_height = img_height
        self.img_width = img_width
        self.labels = []  # List to store encoded labels
        self.images = []  # List to store image data (initially as list of arrays, then concatenated)

        print("Loading frac/width/orientation-conditioned dataset...")
        num_files_loaded = 0
        num_images_total = 0

        # Iterate through all combinations of frac, width, and orientation values.
        for frac in self.frac_values:
            images_for_frac_width_orient = [] # Temporary list to hold images for the current frac (and all its widths/orients)
            labels_for_frac_width_orient = [] # Temporary list for corresponding labels

            for width in self.width_values:
                for orient in self.orientation_values:
                    # Construct the expected filename.
                    filename = self.filename_pattern.format(frac=frac, width=width, orient=orient)
                    filepath = os.path.join(self.data_dir, filename)

                    if os.path.exists(filepath):
                        try:
                            # Load the .npy file.
                            data_raw = np.load(filepath)

                            # Reshape data if necessary. The goal is to have (N, H, W) format.
                            # N = number of images in the file, H = height, W = width.
                            if data_raw.ndim == 3 and data_raw.shape[0] == self.img_height and data_raw.shape[1] == self.img_width:
                                # Assumes (H, W, N) format, transpose to (N, H, W).
                                data = np.transpose(data_raw, (2, 0, 1))
                            elif data_raw.ndim == 3 and data_raw.shape[1] == self.img_height and data_raw.shape[2] == self.img_width:
                                # Assumes (N, H, W) format already.
                                data = data_raw
                            elif data_raw.ndim == 2 and data_raw.shape[0] == self.img_height and data_raw.shape[1] == self.img_width:
                                # Assumes a single image (H, W), add a batch dimension to make it (1, H, W).
                                data = data_raw[np.newaxis, :, :]
                            else:
                                print(f"  WARNING: Skipping {filename}. Unexpected raw shape {data_raw.shape}. Expected H={self.img_height}, W={self.img_width}.")
                                continue

                            num_images_in_file = data.shape[0]
                            if num_images_in_file == 0:
                                print(f"  Warning: File {filename} loaded but contains 0 images.")
                                continue

                            # Append loaded image data (as float32)
                            images_for_frac_width_orient.append(data.astype(np.float32))
                            # Encode the label for this file's parameters.
                            encoded_label = encode_labels(frac, width, orient)
                            # Extend the label list, repeating the label for each image in the current file.
                            labels_for_frac_width_orient.extend([encoded_label] * num_images_in_file)

                            num_files_loaded += 1
                            num_images_total += num_images_in_file
                        except Exception as e:
                            print(f"  ERROR loading or processing {filepath}: {e}")
                            traceback.print_exc() # Print full error trace for debugging.
                    # else:
                    #     print(f"  File not found: {filepath}") # Optional: uncomment for debugging missing files

            # After processing all widths and orientations for a given frac:
            if images_for_frac_width_orient:
                # Concatenate all images collected for this specific 'frac' value.
                self.images.append(np.concatenate(images_for_frac_width_orient, axis=0))
                # Extend the main labels list with labels collected for this 'frac'.
                self.labels.extend(labels_for_frac_width_orient)

        # After iterating through all frac values:
        if not self.images:
            # If no images were loaded at all, raise an error.
            raise FileNotFoundError(f"No data files found or loaded successfully. Check DATA_DIR and FILENAME_PATTERN.")

        # Concatenate all collected image arrays into a single NumPy array.
        self.images = np.concatenate(self.images, axis=0)
        # Add a channel dimension: (N, H, W) -> (N, 1, H, W) for grayscale images.
        self.images = np.expand_dims(self.images, axis=1)
        # Stack all encoded label tensors into a single tensor.
        self.labels = torch.stack(self.labels)

        # Print summary statistics about the loaded dataset.
        print("-" * 30)
        print(f"Dataset loaded successfully.")
        print(f"Total files loaded: {num_files_loaded}")
        print(f"Total images: {len(self.images)}")
        print(f"Image shape: {self.images.shape}") # Should be (N, C, H, W), e.g., (num_images, 1, 128, 128)
        print(f"Labels shape: {self.labels.shape}") # Should be (N, LABEL_DIM), e.g., (num_images, 4)
        print("-" * 30)
        if len(self.labels) > 0:
            print("Example encoded label (first item):", self.labels[0].numpy())
            dec_frac, dec_width, dec_orient = decode_labels(self.labels[0])
            print(f"Decoded: frac={dec_frac:.1f}, width={dec_width:.1f}, orient={dec_orient:.1f} deg")
        else:
            print("WARNING: No labels loaded, dataset might be empty.")
        print("-" * 30)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the given index.
        The image is normalized to the [-1, 1] range, which is common for GANs
        that use tanh activation in the generator's output layer.
        """
        # Get the image at idx, convert to PyTorch tensor.
        # self.images is already (N, C, H, W) and float32.
        image = torch.from_numpy(self.images[idx])
        # Normalize pixel values from [0, 1] (assuming they are) to [-1, 1].
        # If original data is not [0,1], this normalization needs adjustment.
        # Assuming .npy files store values like 0-255 or 0-1. If 0-255, divide by 255 first.
        # For this code, it's assumed the .npy data is implicitly [0,1] or can be treated as such before this step.
        image = image * 2.0 - 1.0
        # Get the corresponding label.
        label = self.labels[idx]
        return image, label

# --- Self-Attention Module ---
class Self_Attn(nn.Module):
    """
    Self-attention Layer (related to SAGAN - Self-Attention Generative Adversarial Networks).
    This module allows the model to weigh the importance of different spatial regions
    when generating or classifying features, capturing long-range dependencies.
    """
    def __init__(self, in_dim, use_spectral_norm=False):
        """
        Args:
            in_dim (int): Number of input channels (feature maps).
            use_spectral_norm (bool): Whether to apply spectral normalization to the conv layers.
                                      Often used in GAN discriminators/critics.
        """
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim # Store input channel dimension.

        # Define a lambda for creating convolutional layers.
        # This makes it easy to toggle spectral normalization.
        conv_layer = lambda C_in, C_out, K, S, P, B: nn.Conv2d(C_in, C_out, K, S, P, bias=B)
        if use_spectral_norm:
            # If true, wrap the Conv2d layer with spectral_norm.
            conv_layer = lambda C_in, C_out, K, S, P, B: spectral_norm(nn.Conv2d(C_in, C_out, K, S, P, bias=B))

        # 1x1 Convolution to generate query vectors.
        # Output channels are in_dim // 8 to reduce computational cost.
        self.query_conv = conv_layer(in_dim, in_dim // 8, 1, 1, 0, False)
        # 1x1 Convolution to generate key vectors.
        self.key_conv = conv_layer(in_dim, in_dim // 8, 1, 1, 0, False)
        # 1x1 Convolution to generate value vectors.
        # Output channels remain in_dim as these are the features to be re-weighted.
        self.value_conv = conv_layer(in_dim, in_dim, 1, 1, 0, False)

        # Learnable scalar parameter 'gamma', initialized to 0.
        # This allows the network to initially bypass the attention mechanism (residual connection)
        # and gradually learn to incorporate it.
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax function to normalize attention scores. Applied across the keys for each query.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for the self-attention module.
        Args:
            x (torch.Tensor): Input feature map of shape (batch_size, C, width, height).
        Returns:
            torch.Tensor: Output feature map with attention applied, same shape as input.
        """
        m_batchsize, C, width, height = x.size()

        # Project input 'x' into query, key, and value spaces.
        # proj_query: (B, N, C') where N = width*height, C' = in_dim//8
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key: (B, C', N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # Calculate attention energy (similarity) by matrix multiplying queries and keys.
        # energy: (B, N, N) - similarity of each pixel (query) to every other pixel (key).
        energy = torch.bmm(proj_query, proj_key)
        # Normalize energy to get attention weights.
        # attention: (B, N, N) - weights sum to 1 for each row (query).
        attention = self.softmax(energy)

        # proj_value: (B, C, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # Compute the attended output by matrix multiplying values with transposed attention weights.
        # out: (B, C, N) - weighted sum of values based on attention.
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # Reshape 'out' back to the original image-like format (B, C, width, height).
        out = out.view(m_batchsize, C, width, height)

        # Apply the learned gamma parameter and add the original input (residual connection).
        # This allows the model to control how much of the attention output is used.
        out = self.gamma * out + x
        return out

# --- Define Networks (with Self-Attention) ---

def weights_init(m):
    """
    Custom weights initialization function.
    Initializes weights of Conv2d, ConvTranspose2d, BatchNorm2d, and Linear layers.
    This can help with training stability and convergence.
    """
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Initialize weights from a normal distribution with mean 0, std 0.02.
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm weights from a normal distribution (mean 1, std 0.02) and biases to 0.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # Initialize Linear layer weights from a normal distribution.
         nn.init.normal_(m.weight.data, 0.0, 0.02)
         if m.bias is not None:
             # Initialize biases to 0 if they exist.
             nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Conditional Generator Network (incorporating Self-Attention).
    Takes a noise vector 'z' and a label vector as input, and outputs an image.
    Uses PixelShuffle for upsampling.
    """
    def __init__(self, z_dim, label_dim, img_channels, features_g, embed_size, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size # Store target image size.

        # Small neural network to embed the input labels.
        # This projects the raw label_dim into a higher-dimensional embed_size space.
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, embed_size),
            nn.ReLU(True) # ReLU activation.
        )

        # The actual input to the main generator path will be concatenated noise and embedded labels.
        combined_input_dim = z_dim + embed_size

        # Project the combined input to a suitable size for reshaping into initial feature maps.
        # Here, it's projected to features_g * 16 channels for a 4x4 feature map.
        # (features_g * 16) * 4 * 4 = features_g * 256
        self.project = nn.Sequential(
            nn.Linear(combined_input_dim, features_g * 16 * 4 * 4),
            nn.ReLU(True)
        )

        # Upsampling Path using Conv2d + PixelShuffle
        # PixelShuffle(r) reshapes (B, C*r^2, H, W) to (B, C, H*r, W*r).
        # So, Conv2d must output C*r^2 channels. Here r=2.

        # Block 1: 4x4 -> 8x8
        # Input: B x (features_g*16) x 4 x 4
        # Conv2d outputs features_g * 8 * (2^2) = features_g * 32 channels
        self.up1 = nn.Sequential(
            nn.Conv2d(features_g * 16, features_g * 8 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 8 * 4), # Batch normalization
            nn.ReLU(True),                     # ReLU activation
            nn.PixelShuffle(2)                 # Upsample by factor of 2. Output: B x (features_g*8) x 8 x 8
        )
        # Block 2: 8x8 -> 16x16
        # Input: B x (features_g*8) x 8 x 8
        self.up2 = nn.Sequential(
            nn.Conv2d(features_g * 8, features_g * 4 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4 * 4),
            nn.ReLU(True),
            nn.PixelShuffle(2)                 # Output: B x (features_g*4) x 16x16
        )
        # Block 3: 16x16 -> 32x32
        # Input: B x (features_g*4) x 16x16
        self.up3 = nn.Sequential(
            nn.Conv2d(features_g * 4, features_g * 2 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2 * 4),
            nn.ReLU(True),
            nn.PixelShuffle(2)                 # Output: B x (features_g*2) x 32x32
        )

        # Block 4: 32x32 -> 64x64
        # Input: B x (features_g*2) x 32x32
        self.up4 = nn.Sequential(
            nn.Conv2d(features_g * 2, features_g * 1 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 1 * 4),
            nn.ReLU(True),
            nn.PixelShuffle(2)                 # Output: B x features_g x 64x64
        )

        # Self-Attention layer applied at 64x64 resolution with 'features_g' channels.
        # Spectral norm is typically not used in the generator's attention layer.
        self.attn1 = Self_Attn(features_g, use_spectral_norm=False)

        # Output Convolution Block: 64x64 -> 128x128
        # Input: B x features_g x 64x64
        # Conv2d outputs img_channels * (2^2) channels for PixelShuffle.
        self.output_conv = nn.Sequential(
            nn.Conv2d(features_g, img_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),                # Output: B x img_channels x 128x128
            nn.Tanh()                          # Tanh activation to scale output to [-1, 1].
        )

    def forward(self, noise, labels):
        """
        Forward pass for the Generator.
        Args:
            noise (torch.Tensor): Latent noise vector, shape (batch_size, Z_DIM).
            labels (torch.Tensor): Encoded label vector, shape (batch_size, LABEL_DIM).
        Returns:
            torch.Tensor: Generated image, shape (batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).
        """
        # Embed the labels.
        lbl_emb = self.label_emb(labels) # Shape: (batch_size, EMBED_SIZE)
        # Concatenate noise and embedded labels along the feature dimension.
        gen_input = torch.cat((noise, lbl_emb), 1) # Shape: (batch_size, Z_DIM + EMBED_SIZE)

        # Project and reshape to initial 4x4 feature maps.
        out = self.project(gen_input)
        out = out.view(out.size(0), -1, 4, 4) # Reshape to B x (features_g*16) x 4 x 4

        # Pass through upsampling blocks.
        out = self.up1(out) # B x (features_g*8) x 8x8
        out = self.up2(out) # B x (features_g*4) x 16x16
        out = self.up3(out) # B x (features_g*2) x 32x32
        out = self.up4(out) # B x features_g x 64x64

        # Apply self-attention.
        out = self.attn1(out) # B x features_g x 64x64

        # Generate final image.
        img = self.output_conv(out) # B x img_channels x 128x128
        return img

class Critic(nn.Module):
    """
    Conditional Critic Network (Discriminator for WGAN-GP, incorporating Self-Attention and Projection Discriminator).
    Takes an image and a label as input, and outputs a scalar score (criticism).
    Uses spectral normalization for stability.
    Implements Projection Discriminator architecture for conditioning.
    """
    def __init__(self, img_channels, label_dim, features_d, embed_size, img_size=128):
        super(Critic, self).__init__()
        self.img_size = img_size

        # Linear layer to embed labels for the projection part of the discriminator.
        # This embedding will be used to compute the conditional part of the score.
        self.label_embedding = nn.Linear(label_dim, embed_size) # Output: B x embed_size

        # Main convolutional path for downsampling the image.
        # All Conv2d layers use spectral normalization.

        # Block 1: 128x128 -> 64x64
        # Input: B x img_channels x 128 x 128
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True) # Leaky ReLU activation. Output: B x features_d x 64x64
        )
        # Block 2: 64x64 -> 32x32
        # Input: B x features_d x 64x64
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True) # Output: B x (features_d*2) x 32x32
        )

        # Self-Attention layer applied at 32x32 resolution with 'features_d*2' channels.
        # Spectral norm is used within the attention module's conv layers.
        self.attn1 = Self_Attn(features_d * 2, use_spectral_norm=True)

        # Block 3: 32x32 -> 16x16 (after attention)
        # Input: B x (features_d*2) x 32x32
        self.down3 = nn.Sequential(
            spectral_norm(nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True) # Output: B x (features_d*4) x 16x16
        )
        # Block 4: 16x16 -> 8x8
        # Input: B x (features_d*4) x 16x16
        self.down4 = nn.Sequential(
            spectral_norm(nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True) # Output: B x (features_d*8) x 8x8
        )
        # Block 5: 8x8 -> 4x4
        # Input: B x (features_d*8) x 8x8
        self.down5 = nn.Sequential(
            spectral_norm(nn.Conv2d(features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True) # Output: B x (features_d*16) x 4x4
        )

        # Store the number of features at the 4x4 stage, used for projection.
        feature_dim_at_4x4 = features_d * 16

        # Output layer for the unconditional part of the critic's score.
        # This is a 4x4 convolution reducing to a single channel, effectively giving one score per patch, then averaged.
        # Input: B x (features_d*16) x 4x4
        self.critic_out_unconditional = spectral_norm(nn.Conv2d(feature_dim_at_4x4, 1, kernel_size=4, stride=1, padding=0, bias=False))
        # Output: B x 1 x 1 x 1

        # Linear layer (with spectral norm) for the conditional part (Projection Discriminator).
        # It projects the embedded label to match the dimensionality of the image features (feature_dim_at_4x4).
        self.label_projection = spectral_norm(nn.Linear(embed_size, feature_dim_at_4x4, bias=False))
        # Output: B x feature_dim_at_4x4

    def forward(self, img, labels):
        """
        Forward pass for the Critic.
        Args:
            img (torch.Tensor): Input image, shape (batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).
            labels (torch.Tensor): Encoded label vector, shape (batch_size, LABEL_DIM).
        Returns:
            torch.Tensor: Scalar critic score for each image in the batch, shape (batch_size,).
        """
        # Pass image through downsampling blocks.
        features = self.down1(img)      # B x features_d x 64x64
        features = self.down2(features)  # B x (features_d*2) x 32x32

        # Apply self-attention.
        # Corrected order: Attention applied on 32x32 features, then further downsampling.
        features = self.attn1(features)  # B x (features_d*2) x 32x32

        # Continue downsampling.
        features = self.down3(features)  # B x (features_d*4) x 16x16
        features = self.down4(features)  # B x (features_d*8) x 8x8
        features = self.down5(features)  # Shape: B x (features_d*16) x 4x4. Let this be 'phi(x)'

        # --- Unconditional part of the score ---
        # This score comes directly from the image features.
        unconditional_score_map = self.critic_out_unconditional(features) # Shape: B x 1 x 1 x 1
        unconditional_score = unconditional_score_map.view(-1) # Flatten to Shape: B (scalar score per image)

        # --- Conditional part of the score (Projection Discriminator) ---
        # This score measures the compatibility between image features and labels.
        # 1. Embed the labels.
        label_emb = self.label_embedding(labels) # Shape: B x embed_size
        # 2. Project embedded labels to the same dimensionality as image features ('phi(x)').
        #    The label_projection layer has weights V. This computes V*embed(y).
        projected_label_emb = self.label_projection(label_emb) # Shape: B x feature_dim_at_4x4

        # 3. Reshape projected labels to be (B, C', 1, 1) to allow broadcasting with image features (B, C', H, W).
        projected_label_reshaped = projected_label_emb.view(projected_label_emb.size(0), -1, 1, 1)

        # 4. Compute the inner product (element-wise multiplication then sum)
        #    between image features 'phi(x)' and projected_label_reshaped.
        #    This is effectively sum_{spatial_dims} ( phi(x) * V*embed(y) )
        #    which is equivalent to (sum_{spatial_dims} phi(x)) . (V*embed(y)) if V*embed(y) is broadcast.
        #    Or, more precisely, it's sum_c sum_h sum_w (phi(x)_{c,h,w} * (V*embed(y))_c )
        #    Here, we do element-wise product and then sum over all dimensions (channel, height, width).
        conditional_term_map = features * projected_label_reshaped # Element-wise product. Result: B x feature_dim_at_4x4 x 4x4
        conditional_score = conditional_term_map.sum(dim=[1,2,3]) # Sum over C', H, W. Result: Shape B

        # Total score is the sum of unconditional and conditional parts.
        output = unconditional_score + conditional_score
        return output


# --- Gradient Penalty Calculation ---
def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    Calculates the gradient penalty for WGAN-GP.
    The penalty enforces the 1-Lipschitz constraint on the critic, which is crucial for
    stable training and for the critic's loss to be a meaningful approximation of the
    Wasserstein distance.
    Args:
        critic (nn.Module): The critic network.
        real_samples (torch.Tensor): Batch of real images.
        fake_samples (torch.Tensor): Batch of fake images generated by the generator.
        labels (torch.Tensor): Batch of labels corresponding to the samples.
                               Must match the batch size of real/fake samples used for interpolation.
        device (torch.device): The device (CPU/GPU) to perform computations on.
    Returns:
        torch.Tensor: The calculated gradient penalty (a scalar tensor).
    """
    batch_size = real_samples.size(0)

    # Ensure labels match the batch size for interpolation. This is important if dataloader drops last batch
    # and batch_size_actual < BATCH_SIZE. The labels passed should correspond to the real_samples.
    if labels.size(0) != batch_size:
        # This case should ideally not be hit if labels are passed correctly from the training loop,
        # corresponding to the current batch of real_samples.
        # If labels provided were for a larger, original batch_size, slice them.
        print(f"Warning: Mismatch in GP label size ({labels.size(0)}) and sample batch size ({batch_size}). Slicing labels.")
        labels_for_interp = labels[:batch_size]
    else:
        labels_for_interp = labels

    # Generate random interpolation factors 'alpha' (epsilon in some papers).
    # Shape: (batch_size, 1, 1, 1) for broadcasting with images.
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Create interpolated samples: x_hat = alpha * x_real + (1 - alpha) * x_fake.
    # These samples lie on straight lines between pairs of real and fake samples.
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Get critic scores for these interpolated samples.
    # Labels are passed as conditioning information.
    critic_interpolates = critic(interpolates, labels_for_interp)

    # Create a tensor of ones for grad_outputs. This is standard for scalar losses.
    fake_grad_outputs = torch.ones_like(critic_interpolates, requires_grad=False)

    # Compute gradients of critic_interpolates with respect to the interpolates.
    # create_graph=True, retain_graph=True are needed because this is part of the critic's loss
    # computation, and we'll backpropagate through this penalty.
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True, # Only get gradients w.r.t. 'interpolates'
    )[0] # autograd.grad returns a tuple, we need the first element.

    # Reshape gradients from (batch_size, C, H, W) to (batch_size, -1) to compute norm per sample.
    gradients = gradients.view(batch_size, -1)

    # Calculate the L2 norm of the gradients for each interpolated sample.
    # The penalty encourages this norm to be 1.
    # (||nabla_x_hat D(x_hat)||_2 - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- Training Setup ---
try:
    print(f"Using device: {DEVICE}")
    print(f"FRAC_VALUES being used: {FRAC_VALUES}") # Verify values after potential corrections.

    # Initialize the custom dataset.
    dataset = FracWidthOrientImageDataset(
        data_dir=DATA_DIR,
        frac_values=FRAC_VALUES,
        width_values=WIDTH_VALUES,
        orientation_values=ORIENTATION_VALUES,
        filename_pattern=FILENAME_PATTERN,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )

    # Check if dataset is empty, which would cause errors later.
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check data loading logic, DATA_DIR, and FILENAME_PATTERN.")

    # Create DataLoader to load data in batches.
    # shuffle=True: shuffles data at every epoch.
    # num_workers: for multi-process data loading. 0 means data loaded in main process.
    # pin_memory=True: speeds up CPU-to-GPU data transfer if using CUDA.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

except Exception as e:
    # Catch any errors during dataset initialization and print details.
    print(f"FATAL: Error initializing dataset: {e}")
    traceback.print_exc() # Print full stack trace.
    exit() # Terminate script if dataset fails to load.

# Initialize Generator and Critic networks.
netG = Generator(Z_DIM, LABEL_DIM, IMG_CHANNELS, G_FEATURES, EMBED_SIZE, IMG_HEIGHT).to(DEVICE)
netC = Critic(IMG_CHANNELS, LABEL_DIM, D_FEATURES, EMBED_SIZE, IMG_HEIGHT).to(DEVICE)

# Apply custom weights initialization to both networks.
netG.apply(weights_init)
netC.apply(weights_init)
print("Applied custom weight initialization.")
# Optional: Print model summaries to check architecture.
# print("Generator summary:\n", netG)
# print("Critic summary:\n", netC)


# Optimizers: Adam is commonly used for GANs.
# Betas (0.0, 0.9) are often recommended for WGAN-GP instead of default (0.9, 0.999).
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizerC = optim.Adam(netC.parameters(), lr=LR, betas=(BETA1, BETA2))

# --- Fixed noise/labels for Visualization ---
# Using fixed noise and labels allows for consistent visualization of the generator's
# progress over time. We can see how images for specific conditions improve.

# Define a list of (frac, width, orientation) triplets for visualization.
fixed_vis_triplets = [
    (FRAC_VALUES[0], WIDTH_VALUES[0], ORIENTATION_VALUES[0]), # e.g., min frac, min width, min orient
    (FRAC_VALUES[-1], WIDTH_VALUES[-1], ORIENTATION_VALUES[-1]),# e.g., max frac, max width, max orient
    (FRAC_VALUES[len(FRAC_VALUES)//2], WIDTH_VALUES[len(WIDTH_VALUES)//2], ORIENTATION_VALUES[len(ORIENTATION_VALUES)//2]), # Mid values
    (FRAC_VALUES[0], WIDTH_VALUES[-1], ORIENTATION_VALUES[0]),   # Mix
    (FRAC_VALUES[-1], WIDTH_VALUES[0], ORIENTATION_VALUES[-1]),  # Mix
    (FRAC_VALUES[0], WIDTH_VALUES[0], ORIENTATION_VALUES[-1]),  # Mix
    (FRAC_VALUES[-1], WIDTH_VALUES[-1], ORIENTATION_VALUES[0])   # Mix
]
num_fixed_samples = len(fixed_vis_triplets) # Number of samples to generate for visualization.
print(f"Setting up fixed visualization for {num_fixed_samples} specific (frac, width, orient) triplets:")
print(fixed_vis_triplets)

# Generate a batch of fixed noise vectors (once).
fixed_noise = torch.randn(num_fixed_samples, Z_DIM, device=DEVICE)
# Encode the fixed label triplets.
fixed_labels_list = [encode_labels(f, w, o) for f, w, o in fixed_vis_triplets]
# Stack them into a tensor and move to device.
fixed_labels = torch.stack(fixed_labels_list).to(DEVICE)
# Store denormalized labels for titling the visualization.
fixed_labels_denorm = [(f, w, o) for f, w, o in fixed_vis_triplets]


# Lists for tracking progress (losses, generated images).
G_losses = []  # Stores Generator loss per generator iteration.
C_losses = []  # Stores Critic loss (average over critic iterations, including GP) per generator iteration.
GP_losses = [] # Stores Gradient Penalty term per generator iteration.
img_list = []  # Stores grids of generated images for creating an animation later (optional).

# --- WGAN-GP Training Loop ---
print("Starting WGAN-GP Training Loop...")
total_gen_iters = 0 # Counter for total generator updates.

for epoch in range(NUM_EPOCHS):
    # Loop over batches of data from the dataloader.
    for i, data in enumerate(dataloader, 0):
        real_images, real_labels = data[0].to(DEVICE), data[1].to(DEVICE)
        batch_size_actual = real_images.size(0) # Actual batch size (can be smaller for last batch).

        # Defensive check, though dataloader should not yield empty batches if dataset is non-empty.
        if batch_size_actual == 0:
            print(f"Warning: Batch {i} in epoch {epoch+1} is empty, skipping.")
            continue

        # --- Train Critic ---
        # The Critic is trained CRITIC_ITERATIONS times for each Generator update.
        mean_iteration_critic_loss = 0 # To store average critic loss over its iterations.
        current_gp_loss_val = 0.0      # To store the GP value from the last critic iteration for logging.

        for critic_iter in range(CRITIC_ITERATIONS):
            netC.zero_grad() # Clear gradients for the critic.

            # Generate a batch of fake images.
            # Noise is sampled randomly for each critic iteration.
            noise = torch.randn(batch_size_actual, Z_DIM, device=DEVICE)
            # Generate fake images conditioned on the real_labels from the current batch.
            # This ensures the critic is learning to distinguish real vs fake *for relevant conditions*.
            # .detach() is used because we are training the Critic, so we don't want gradients
            # to flow back to the Generator at this stage.
            with torch.no_grad(): # Explicitly state no grads for G here.
                fake_images = netG(noise, real_labels).detach()

            # Get critic scores for real images.
            critic_real = netC(real_images, real_labels)
            # Get critic scores for fake images.
            critic_fake = netC(fake_images, real_labels) # Use same real_labels for fake image conditioning.

            # Calculate Gradient Penalty.
            # .data is used for real_images and fake_images to pass tensors without gradient history
            # that is not needed by compute_gradient_penalty for these inputs.
            gradient_penalty = compute_gradient_penalty(netC, real_images.data, fake_images.data, real_labels.data, DEVICE)
            current_gp_loss_val = LAMBDA_GP * gradient_penalty # Store for logging.

            # Critic Loss for WGAN-GP: D_loss = E[D(G(z,y))] - E[D(x,y)] + lambda_gp * GP
            # We want to *minimize* this loss, which means maximizing (E[D(x,y)] - E[D(G(z,y))]).
            loss_C = critic_fake.mean() - critic_real.mean() + current_gp_loss_val
            loss_C.backward() # Compute gradients for the critic.
            optimizerC.step() # Update critic's weights.

            # Accumulate critic loss for averaging.
            mean_iteration_critic_loss += loss_C.item()

        # Average critic loss over CRITIC_ITERATIONS.
        mean_iteration_critic_loss /= CRITIC_ITERATIONS
        C_losses.append(mean_iteration_critic_loss) # Log average critic loss for this generator step.
        GP_losses.append(current_gp_loss_val.item()) # Log GP from the last critic update.

        # --- Train Generator ---
        # Generator is trained once after CRITIC_ITERATIONS of critic training.
        netG.zero_grad() # Clear gradients for the generator.

        # Generate a new batch of noise for this generator update.
        noise_g = torch.randn(batch_size_actual, Z_DIM, device=DEVICE)
        # Generate fake images conditioned on the same real_labels distribution.
        # This makes the generator try to fool the critic for the conditions present in the current batch.
        fake_images_g = netG(noise_g, real_labels)

        # Get critic scores for these new fake images.
        # We want the generator to produce images that the critic thinks are real.
        critic_fake_g = netC(fake_images_g, real_labels)

        # Generator Loss: G_loss = -E[D(G(z,y))]
        # The generator tries to *maximize* E[D(G(z,y))], so we *minimize* its negative.
        loss_G = -critic_fake_g.mean()
        loss_G.backward() # Compute gradients for the generator.
        optimizerG.step() # Update generator's weights.
        G_losses.append(loss_G.item()) # Log generator loss.
        total_gen_iters += 1

        # --- Logging & Checkpointing ---
        # Log progress every 100 generator iterations.
        if total_gen_iters % 100 == 0:
             # For more accurate Wasserstein distance estimation logging:
             # Re-calculate critic scores on current real and newly generated fake images
             # without involving gradient penalty or optimizer steps.
             with torch.no_grad():
                noise_eval = torch.randn(batch_size_actual, Z_DIM, device=DEVICE)
                fake_images_eval = netG(noise_eval, real_labels).detach()
                critic_real_eval_mean = netC(real_images, real_labels).mean()
                critic_fake_eval_mean = netC(fake_images_eval, real_labels).mean()
                # Wasserstein distance estimate: E[D(x,y)] - E[D(G(z,y))]
                # A higher (less negative or more positive) value is generally better.
                w_dist = critic_real_eval_mean - critic_fake_eval_mean
             print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i+1}/{len(dataloader)}] | GenIter [{total_gen_iters}] | '
                   f'Loss_G: {loss_G.item():.4f} | AvgLoss_C_with_GP: {mean_iteration_critic_loss:.4f} | '
                   f'Last_GP_Term: {GP_losses[-1]:.4f} | W-Dist_est: {w_dist.item():.4f}')

        # Save generated images and model checkpoints every 500 generator iterations,
        # and also at the very end of training.
        if total_gen_iters % 500 == 0 or \
           (epoch == NUM_EPOCHS - 1 and i == len(dataloader)-1 and total_gen_iters > 0):
            print(f"--- Checkpointing at Gen Iteration {total_gen_iters} ---")
            with torch.no_grad(): # No gradients needed for generation.
                netG.eval() # Set generator to evaluation mode.
                # Generate images using the fixed noise and labels for consistent visualization.
                fixed_fake_samples = netG(fixed_noise, fixed_labels).detach().cpu()
                netG.train() # Set generator back to training mode.

            # Create a grid of the generated fixed samples.
            # `normalize=True` scales images to [0,1] for display.
            vis_nrow = min(num_fixed_samples, 8) # Number of images per row in the grid.
            img_grid = vutils.make_grid(fixed_fake_samples, padding=2, normalize=True, nrow=vis_nrow)
            img_list.append(img_grid) # Store for potential animation.

            # Plot and save the image grid.
            fig = plt.figure(figsize=(vis_nrow * 2, math.ceil(num_fixed_samples/vis_nrow) * 2.5)) # Adjust figure size.
            plt.axis("off") # Turn off axes for cleaner image display.
            # Create a title string with the (F,W,O) values for each image in the grid.
            title_labels_str_list = []
            for f_val, w_val, o_val in fixed_labels_denorm:
                 title_labels_str_list.append(f"({f_val:.0f},{w_val:.0f},{o_val:.0f}°)")
            # Format title to fit multiple lines if many samples.
            title_labels_str = "\n".join([", ".join(title_labels_str_list[k:k+vis_nrow])
                                         for k in range(0, len(title_labels_str_list), vis_nrow)])
            plt.title(f"GenIter {total_gen_iters}\nFixed (Frac,Width,Orient):\n{title_labels_str}", fontsize=8)

            plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)), cmap='gray') # Transpose from (C,H,W) to (H,W,C) for imshow.
            save_path_img = os.path.join(IMG_DIR, f"frac_width_orient_fixed_gen_iter_{total_gen_iters:06d}.png")
            plt.savefig(save_path_img)
            print(f"Saved visualization: {save_path_img}")
            plt.close(fig) # Close figure to free memory.

            # Save model checkpoints.
            save_path_g = os.path.join(MODEL_DIR, f'netG_frac_width_orient_iter_{total_gen_iters:06d}.pth')
            save_path_c = os.path.join(MODEL_DIR, f'netC_frac_width_orient_iter_{total_gen_iters:06d}.pth')
            torch.save(netG.state_dict(), save_path_g) # Save generator state dictionary.
            torch.save(netC.state_dict(), save_path_c) # Save critic state dictionary.
            print(f"Saved models: {save_path_g}, {save_path_c}")

print("--- Training Finished ---")

# --- Plotting Losses (Optional) ---
# Plot Generator, Critic, and Gradient Penalty losses over training iterations.
plt.figure(figsize=(12, 6))
plt.title("Generator, Critic (avg w/ GP), and GP Term Losses During Training")
plt.plot(G_losses, label="Generator Loss (G)")
plt.plot(C_losses, label="Critic Loss (C_avg_with_GP)")
plt.plot(GP_losses, label="Gradient Penalty Term (GP_term)")
plt.xlabel("Generator Iterations")
plt.ylabel("Loss Value")
plt.legend()
loss_plot_path = os.path.join(OUTPUT_DIR, "loss_plot_final.png")
plt.savefig(loss_plot_path)
print(f"Saved loss plot to: {loss_plot_path}")
plt.show()

# If you want to create an animation from the saved img_list (e.g., a GIF):
# import imageio
# if img_list:
#     ims = [np.transpose(i.numpy(), (1, 2, 0)) for i in img_list]
#     imageio.mimsave(os.path.join(OUTPUT_DIR, 'generation_animation.gif'), ims, fps=5)
#     print("Saved generation animation.")