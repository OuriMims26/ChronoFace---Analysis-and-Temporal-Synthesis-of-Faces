import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import time
import datetime
import sys
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image, make_grid


# ============================================================================
# WEIGHT INITIALIZATION UTILITIES
# ============================================================================

def initialize_network_weights(module):
    """
    Initialize network parameters with normal distribution.

    This function applies proper weight initialization to convolutional
    and batch normalization layers, which is crucial for stable training
    and faster convergence in GANs.

    Initialization strategy:
        - Convolutional layers: Normal(mean=0.0, std=0.02)
        - Batch normalization: weights Normal(mean=1.0, std=0.02), bias = 0

    Args:
        module: PyTorch layer/module to initialize
    """
    # Get the class name of the module (e.g., "Conv2d", "BatchNorm2d")
    layer_type = module.__class__.__name__

    # Initialize convolutional layers
    if layer_type.find("Conv") != -1:
        # Initialize weights with small random values centered at 0
        # Standard deviation of 0.02 prevents gradient explosion/vanishing
        torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

        # Initialize bias to zero if it exists
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)

    # Initialize batch normalization layers
    elif layer_type.find("BatchNorm2d") != -1:
        # Weight (gamma) initialized near 1.0 for identity mapping initially
        torch.nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        # Bias (beta) initialized to zero
        torch.nn.init.constant_(module.bias.data, 0.0)


# ============================================================================
# REPLAY BUFFER FOR TRAINING STABILITY
# ============================================================================

class ImageHistoryBuffer:
    """
    Maintains a buffer of previously generated images for discriminator training.

    CycleGAN uses a history buffer to prevent discriminators from overfitting
    to the most recent generator outputs. By training on a mix of current and
    past generated images, the discriminator becomes more robust and training
    is more stable.

    The buffer stores up to 'max_capacity' images and randomly decides whether
    to return a stored image or the current one, preventing mode collapse.

    Attributes:
        max_capacity (int): Maximum number of images to store (default: 50)
        storage (list): Internal list maintaining the image history
    """

    def __init__(self, max_capacity=50):
        """
        Initialize the replay buffer.

        Args:
            max_capacity: Maximum number of images to keep in history

        Raises:
            AssertionError: If max_capacity is not positive
        """
        assert max_capacity > 0, "Buffer capacity must be positive"
        self.max_capacity = max_capacity
        self.storage = []

    def insert_and_retrieve(self, input_batch):
        """
        Store new images and return a mix of current/historical images.

        This method implements the core replay buffer logic:
        1. If buffer not full: add image and return it
        2. If buffer full: randomly choose to either:
           - Return a random historical image (50% chance)
           - Return the current image (50% chance)

        This randomization prevents discriminators from only seeing
        the newest generator outputs, improving training stability.

        Args:
            input_batch: Batch of generated images from generator

        Returns:
            Variable: Batch of images (mix of new and historical)
        """
        output_batch = []

        # Process each image in the batch individually
        for img in input_batch.data:
            # Add batch dimension (required for proper concatenation later)
            img = torch.unsqueeze(img, 0)

            # Case 1: Buffer not yet at capacity
            if len(self.storage) < self.max_capacity:
                # Add to buffer and return current image
                self.storage.append(img)
                output_batch.append(img)

            # Case 2: Buffer is full - use replay mechanism
            else:
                # Randomly decide: use historical image or current image
                if random.uniform(0, 1) > 0.5:
                    # Select random index from buffer
                    random_idx = random.randint(0, self.max_capacity - 1)

                    # Return the historical image (cloned to prevent modifications)
                    output_batch.append(self.storage[random_idx].clone())

                    # Replace that historical image with current one
                    self.storage[random_idx] = img
                else:
                    # Return current image without buffering
                    output_batch.append(img)

        # Concatenate all images into a single batch and wrap in Variable
        return Variable(torch.cat(output_batch))


# ============================================================================
# RESIDUAL BLOCK FOR DEEP FEATURE LEARNING
# ============================================================================

class IdentityPreservingBlock(nn.Module):
    """
    Residual block with skip connection for deep networks.

    Residual blocks allow training of very deep networks by providing
    a direct path for gradients through skip connections. This prevents
    the vanishing gradient problem and allows the network to learn
    identity mappings easily.

    Architecture:
        Input → [ReflectionPad → Conv → InstanceNorm → ReLU] × 2 → Add → Output
                  ↓                                                    ↑
                  └────────────────── Skip Connection ─────────────────┘

    Args:
        num_features (int): Number of input/output channels
    """

    def __init__(self, num_features):
        super(IdentityPreservingBlock, self).__init__()

        # Sequential block without skip connection
        # The skip connection is added in the forward() method
        self.conv_block = nn.Sequential(
            # Reflection padding prevents border artifacts (better than zero padding)
            nn.ReflectionPad2d(1),

            # 3x3 convolution maintains spatial dimensions
            nn.Conv2d(num_features, num_features, kernel_size=3),

            # Instance normalization (better than batch norm for style transfer)
            nn.InstanceNorm2d(num_features),

            # ReLU activation with inplace operation (saves memory)
            nn.ReLU(inplace=True),

            # Second convolution block (same structure)
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=3),
            nn.InstanceNorm2d(num_features),
        )

    def forward(self, x):
        """
        Forward pass with residual connection.

        The output is the sum of:
        - Transformed features (learned transformation)
        - Original input (identity mapping)

        This allows the network to learn residual functions F(x)
        instead of the complete mapping H(x), making optimization easier.

        Args:
            x: Input feature map

        Returns:
            Output feature map with same dimensions as input
        """
        return x + self.conv_block(x)


# ============================================================================
# GENERATOR NETWORK (ResNet-based Architecture)
# ============================================================================

class ResidualGenerator(nn.Module):
    """
    Generator network with residual blocks for image-to-image translation.

    Architecture follows the design from the CycleGAN paper:
    - Encoder: Downsamples input to capture high-level features
    - Transformer: Multiple residual blocks for feature transformation
    - Decoder: Upsamples back to original resolution

    This architecture preserves spatial structure while allowing
    significant appearance changes through the residual blocks.

    Args:
        input_dimensions (tuple): Input shape as (channels, height, width)
        num_residual_blocks (int): Number of residual blocks (default: 9)
                                   More blocks = more transformation capacity
    """

    def __init__(self, input_dimensions, num_residual_blocks=9):
        super(ResidualGenerator, self).__init__()

        # Extract number of channels from input shape
        input_channels = input_dimensions[0]

        # ====================================================================
        # INITIAL CONVOLUTION BLOCK
        # ====================================================================
        # Large 7x7 kernel captures broad spatial context

        initial_filters = 64
        network_layers = [
            nn.ReflectionPad2d(input_channels),
            nn.Conv2d(input_channels, initial_filters, kernel_size=7),
            nn.InstanceNorm2d(initial_filters),
            nn.ReLU(inplace=True),
        ]
        current_filters = initial_filters

        # ====================================================================
        # ENCODER: DOWNSAMPLING LAYERS
        # ====================================================================
        # Two downsampling blocks reduce spatial dimensions by 4x
        # While doubling feature channels to capture more abstract features

        for _ in range(2):
            # Double the number of filters
            next_filters = current_filters * 2

            network_layers += [
                # Strided convolution for downsampling (learnable vs pooling)
                nn.Conv2d(current_filters, next_filters,
                          kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(next_filters),
                nn.ReLU(inplace=True),
            ]

            current_filters = next_filters

        # ====================================================================
        # TRANSFORMER: RESIDUAL BLOCKS
        # ====================================================================
        # Residual blocks maintain spatial dimensions while transforming features
        # These are the core of the transformation capability

        for _ in range(num_residual_blocks):
            network_layers += [IdentityPreservingBlock(current_filters)]

        # ====================================================================
        # DECODER: UPSAMPLING LAYERS
        # ====================================================================
        # Mirror the encoder by upsampling back to original resolution

        for _ in range(2):
            # Halve the number of filters
            next_filters = current_filters // 2

            network_layers += [
                # Nearest-neighbor upsampling followed by convolution
                # This approach reduces checkerboard artifacts
                nn.Upsample(scale_factor=2),
                nn.Conv2d(current_filters, next_filters,
                          kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(next_filters),
                nn.ReLU(inplace=True),
            ]

            current_filters = next_filters

        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================
        # Final 7x7 convolution maps to RGB output
        # Tanh activation produces values in [-1, 1] range (standard for GANs)

        network_layers += [
            nn.ReflectionPad2d(input_channels),
            nn.Conv2d(current_filters, input_channels, kernel_size=7),
            nn.Tanh()
        ]

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        """
        Generate translated image from input.

        Args:
            x: Input image tensor [batch, channels, height, width]

        Returns:
            Translated image tensor with same dimensions
        """
        return self.network(x)


# ============================================================================
# DISCRIMINATOR NETWORK (PatchGAN Architecture)
# ============================================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local realism evaluation.

    Unlike traditional discriminators that classify entire images,
    PatchGAN outputs a grid where each value represents the "realness"
    of a corresponding image patch. This approach:
    - Encourages high-frequency detail generation
    - Has fewer parameters than full-image discriminators
    - Is more effective for texture/style consistency

    The receptive field of each output patch is 70x70 pixels,
    meaning each output value evaluates a 70x70 region of the input.

    Args:
        input_dimensions (tuple): Input shape as (channels, height, width)
    """

    def __init__(self, input_dimensions):
        super(PatchDiscriminator, self).__init__()

        input_channels, input_height, input_width = input_dimensions

        # Calculate output dimensions after 4 downsampling layers (each divides by 2)
        # Output is a grid of patch classifications, not a single value
        self.output_dimensions = (1, input_height // 2 ** 4, input_width // 2 ** 4)

        # ====================================================================
        # DISCRIMINATOR ARCHITECTURE
        # ====================================================================
        # Progressive downsampling with increasing feature depth

        self.network = nn.Sequential(
            # First layer: no normalization (standard practice for discriminators)
            *self._create_conv_block(input_channels, 64, apply_normalization=False),

            # Subsequent layers: progressively double filters
            *self._create_conv_block(64, 128),
            *self._create_conv_block(128, 256),
            *self._create_conv_block(256, 512),

            # Zero padding to maintain dimensions for final convolution
            nn.ZeroPad2d((1, 0, 1, 0)),

            # Final 1x1 convolution produces patch-wise classification
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def _create_conv_block(self, input_filters, output_filters,
                           apply_normalization=True):
        """
        Create a discriminator convolutional block.

        Each block performs:
        1. Strided convolution (downsamples by 2x)
        2. Optional instance normalization
        3. LeakyReLU activation (allows small negative gradients)

        Args:
            input_filters (int): Number of input channels
            output_filters (int): Number of output channels
            apply_normalization (bool): Whether to apply instance normalization

        Returns:
            list: Layers comprising the discriminator block
        """
        block_layers = [
            # 4x4 kernel with stride 2 for downsampling
            nn.Conv2d(input_filters, output_filters,
                      kernel_size=4, stride=2, padding=1)
        ]

        # Instance normalization (optional, skipped in first layer)
        if apply_normalization:
            block_layers.append(nn.InstanceNorm2d(output_filters))

        # LeakyReLU prevents dying ReLU problem
        # Negative slope of 0.2 allows small gradients for negative values
        block_layers.append(nn.LeakyReLU(0.2, inplace=True))

        return block_layers

    def forward(self, input_image):
        """
        Evaluate image authenticity as a patch grid.

        Args:
            input_image: Input image tensor [batch, channels, height, width]

        Returns:
            Patch classification grid [batch, 1, height//16, width//16]
            Each value represents realness score for corresponding patch
        """
        return self.network(input_image)