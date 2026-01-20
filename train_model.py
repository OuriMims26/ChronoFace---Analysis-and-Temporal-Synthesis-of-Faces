import os
import itertools
import sys
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

# Import custom modules for dataset handling and model architectures
from data import *
from model import *

# ============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

# Dataset identifier used for organizing outputs
DATASET_ID = 'old2young'

# Create output directories for generated images and model weights
os.makedirs("images/%s" % DATASET_ID, exist_ok=True)
os.makedirs("saved_models/%s" % DATASET_ID, exist_ok=True)

# Training duration parameters
TOTAL_EPOCHS = 10
STARTING_EPOCH = 0  # Set to non-zero value to resume from checkpoint
EPOCH_OFFSET = 0  # Offset for learning rate scheduling when resuming

# Learning rate decay configuration
# After this epoch, learning rate will linearly decay to zero
LR_DECAY_START = 3

# Network input specifications
# Format: (channels, height, width)
INPUT_DIMENSIONS = (3, 40, 40)
NUM_CHANNELS, IMAGE_H, IMAGE_W = INPUT_DIMENSIONS

# Training hyperparameters
BATCH_SIZE = 1  # Number of samples per batch
LEARNING_RATE = 2e-4  # Initial learning rate for Adam optimizer

# Checkpoint and visualization intervals
CHECKPOINT_SAVE_FREQ = 1  # Save model weights every N epochs
SAMPLE_GENERATION_FREQ = 100  # Generate sample images every N batches

# Loss function weights
# These coefficients balance different objectives during training
CYCLE_CONSISTENCY_WEIGHT = 10  # Weight for cycle consistency loss
IDENTITY_PRESERVATION_WEIGHT = 5  # Weight for identity mapping loss

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

# Adversarial loss: measures how well discriminator can distinguish real/fake
adversarial_criterion = torch.nn.MSELoss()

# Cycle consistency loss: ensures reconstructed images match originals
# Uses L1 (Mean Absolute Error) for pixel-wise comparison
cycle_criterion = torch.nn.L1Loss()

# Identity loss: preserves color composition when input already in target domain
identity_criterion = torch.nn.L1Loss()

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Generator networks for bidirectional translation
# G_AB: transforms domain A (young) to domain B (old)
# G_BA: transforms domain B (old) to domain A (young)
generator_A_to_B = GeneratorResNet(INPUT_DIMENSIONS, num_residual_blocks=3)
generator_B_to_A = GeneratorResNet(INPUT_DIMENSIONS, num_residual_blocks=3)

# Discriminator networks to distinguish real from generated images
# D_A: evaluates authenticity of images in domain A
# D_B: evaluates authenticity of images in domain B
discriminator_A = Discriminator(INPUT_DIMENSIONS)
discriminator_B = Discriminator(INPUT_DIMENSIONS)

# Check for GPU availability and move models accordingly
use_cuda = torch.cuda.is_available()

if use_cuda:
    # Transfer all models and loss functions to GPU for faster computation
    generator_A_to_B = generator_A_to_B.cuda()
    generator_B_to_A = generator_B_to_A.cuda()
    discriminator_A = discriminator_A.cuda()
    discriminator_B = discriminator_B.cuda()
    adversarial_criterion.cuda()
    cycle_criterion.cuda()
    identity_criterion.cuda()

# ============================================================================
# WEIGHT INITIALIZATION OR CHECKPOINT LOADING
# ============================================================================

if STARTING_EPOCH != 0:
    # Resume training from saved checkpoints
    # This allows continuation of interrupted training sessions
    print(f"Loading models from epoch {STARTING_EPOCH}...")

    generator_A_to_B.load_state_dict(
        torch.load("saved_models/%s/G_AB_%d.pth" % (DATASET_ID, STARTING_EPOCH))
    )
    generator_B_to_A.load_state_dict(
        torch.load("saved_models/%s/G_BA_%d.pth" % (DATASET_ID, STARTING_EPOCH))
    )
    discriminator_A.load_state_dict(
        torch.load("saved_models/%s/D_A_%d.pth" % (DATASET_ID, STARTING_EPOCH))
    )
    discriminator_B.load_state_dict(
        torch.load("saved_models/%s/D_B_%d.pth" % (DATASET_ID, STARTING_EPOCH))
    )
else:
    # Initialize weights from scratch using normal distribution
    # This helps with gradient flow and training stability
    generator_A_to_B.apply(weights_init_normal)
    generator_B_to_A.apply(weights_init_normal)
    discriminator_A.apply(weights_init_normal)
    discriminator_B.apply(weights_init_normal)

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

# Generator optimizer: jointly optimizes both generators
# Using itertools.chain to combine parameters from both networks
optimizer_generators = torch.optim.Adam(
    itertools.chain(generator_A_to_B.parameters(), generator_B_to_A.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)  # Beta values for Adam momentum
)

# Separate optimizers for each discriminator
optimizer_disc_A = torch.optim.Adam(
    discriminator_A.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_disc_B = torch.optim.Adam(
    discriminator_B.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

# Define learning rate decay function
# Keeps LR constant until LR_DECAY_START, then linearly decreases to 0
def compute_lr_decay(current_epoch):
    """
    Calculate learning rate multiplier for current epoch.

    Returns 1.0 (no decay) before LR_DECAY_START epoch,
    then linearly decreases to 0.0 by the final epoch.
    """
    decay_progress = max(0, current_epoch + EPOCH_OFFSET - LR_DECAY_START)
    decay_range = TOTAL_EPOCHS - LR_DECAY_START
    return 1.0 - (decay_progress / decay_range)


# Create schedulers for all optimizers
scheduler_gen = torch.optim.lr_scheduler.LambdaLR(
    optimizer_generators, lr_lambda=compute_lr_decay
)
scheduler_disc_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_disc_A, lr_lambda=compute_lr_decay
)
scheduler_disc_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_disc_B, lr_lambda=compute_lr_decay
)

# ============================================================================
# UTILITY SETUP
# ============================================================================

# Define tensor type based on GPU availability
TensorType = torch.cuda.FloatTensor if use_cuda else torch.Tensor

# Replay buffers store previously generated fake images
# This prevents discriminators from overfitting to most recent generations
# and stabilizes training dynamics
buffer_fake_A = ReplayBuffer()
buffer_fake_B = ReplayBuffer()

# ============================================================================
# DATA PREPROCESSING PIPELINE
# ============================================================================

# Define image transformations for data augmentation and normalization
preprocessing_pipeline = [
    # Resize with slight upscaling to allow random cropping
    transforms.Resize(int(IMAGE_H * 1.12), Image.BICUBIC),

    # Random crop back to original size (adds spatial variation)
    transforms.RandomCrop((IMAGE_H, IMAGE_W)),

    # Random horizontal flip for additional augmentation
    transforms.RandomHorizontalFlip(),

    # Convert PIL Image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize to [-1, 1] range (standard for GANs)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# ============================================================================
# DATA LOADERS
# ============================================================================

# Training dataset loader
# unaligned=True: images from domains A and B don't need to correspond
dataloader_train = DataLoader(
    CycleGANDataset(
        f'{DATASET_ID}',
        transforms_=preprocessing_pipeline,
        unaligned=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for debugging; increase for faster loading
)

# Validation/test dataset loader
# Used for generating visual samples during training
dataloader_val = DataLoader(
    CycleGANDataset(
        f'{DATASET_ID}',
        transforms_=preprocessing_pipeline,
        unaligned=True,
        mode="test"
    ),
    batch_size=5,  # Generate 5 samples at a time for visualization
    shuffle=True,
    num_workers=0,
)

# Determine device for tensor operations
computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# SAMPLE GENERATION FUNCTION
# ============================================================================

def generate_sample_images(batch_number):
    """
    Generate and save a grid of sample translations from the test set.

    This function creates a visualization showing:
    - Row 1: Real images from domain A
    - Row 2: Fake images in domain B (A→B translation)
    - Row 3: Real images from domain B
    - Row 4: Fake images in domain A (B→A translation)

    Args:
        batch_number (int): Current batch iteration, used for filename
    """
    # Retrieve a batch from validation set
    sample_batch = next(iter(dataloader_val))

    # Set generators to evaluation mode (disables dropout, batchnorm updates)
    generator_A_to_B.eval()
    generator_B_to_A.eval()

    # Extract and move images to appropriate device
    real_images_A = sample_batch["A"].to(computation_device)
    generated_B = generator_A_to_B(real_images_A)

    real_images_B = sample_batch["B"].to(computation_device)
    generated_A = generator_B_to_A(real_images_B)

    # Arrange images in horizontal grids (5 images per row)
    grid_real_A = make_grid(real_images_A, nrow=5, normalize=True)
    grid_real_B = make_grid(real_images_B, nrow=5, normalize=True)
    grid_fake_A = make_grid(generated_A, nrow=5, normalize=True)
    grid_fake_B = make_grid(generated_B, nrow=5, normalize=True)

    # Stack grids vertically to create final visualization
    combined_grid = torch.cat((grid_real_A, grid_fake_B, grid_real_B, grid_fake_A), 1)

    # Save the combined grid as an image file
    output_path = "images/%s/%s.png" % (DATASET_ID, batch_number)
    save_image(combined_grid, output_path, normalize=False)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

for current_epoch in range(STARTING_EPOCH, TOTAL_EPOCHS):
    print(f"Starting Epoch: {current_epoch}")

    for batch_idx, batch_data in enumerate(dataloader_train):

        # ====================================================================
        # PREPARE BATCH DATA
        # ====================================================================

        # Move real images from both domains to computation device
        real_A = batch_data["A"].to(computation_device)
        real_B = batch_data["B"].to(computation_device)

        # Create ground truth labels for adversarial training
        # valid: label for real images (ones)
        # fake: label for generated images (zeros)
        label_real = torch.ones(
            (real_A.size(0), *discriminator_A.output_shape),
            requires_grad=False
        ).to(computation_device)

        label_fake = torch.zeros(
            (real_A.size(0), *discriminator_A.output_shape),
            requires_grad=False
        ).to(computation_device)

        # ====================================================================
        # TRAIN GENERATORS
        # ====================================================================

        # Set generators to training mode
        generator_A_to_B.train()
        generator_B_to_A.train()

        # Reset gradients from previous iteration
        optimizer_generators.zero_grad()

        # --- Identity Loss ---
        # Ensures generator doesn't modify images already in target domain
        # Example: G_BA(real_A) should output real_A if A is already "young"
        identity_loss_A = identity_criterion(generator_B_to_A(real_A), real_A)
        identity_loss_B = identity_criterion(generator_A_to_B(real_B), real_B)
        combined_identity_loss = (identity_loss_A + identity_loss_B) / 2

        # --- Adversarial Loss ---
        # Measures how well generators fool discriminators
        fake_B = generator_A_to_B(real_A)
        adversarial_loss_AB = adversarial_criterion(
            discriminator_B(fake_B), label_real
        )

        fake_A = generator_B_to_A(real_B)
        adversarial_loss_BA = adversarial_criterion(
            discriminator_A(fake_A), label_real
        )

        combined_adversarial_loss = (adversarial_loss_AB + adversarial_loss_BA) / 2

        # --- Cycle Consistency Loss ---
        # Ensures A → B → A reconstruction matches original A
        # and B → A → B reconstruction matches original B
        reconstructed_A = generator_B_to_A(fake_B)
        cycle_loss_A = cycle_criterion(reconstructed_A, real_A)

        reconstructed_B = generator_A_to_B(fake_A)
        cycle_loss_B = cycle_criterion(reconstructed_B, real_B)

        combined_cycle_loss = (cycle_loss_A + cycle_loss_B) / 2

        # --- Combined Generator Loss ---
        # Weighted sum of all generator objectives
        total_generator_loss = (
                combined_adversarial_loss +
                CYCLE_CONSISTENCY_WEIGHT * combined_cycle_loss +
                IDENTITY_PRESERVATION_WEIGHT * combined_identity_loss
        )

        # Backpropagate and update generator weights
        total_generator_loss.backward()
        optimizer_generators.step()

        # ====================================================================
        # TRAIN DISCRIMINATOR A (Domain A Authenticity Evaluator)
        # ====================================================================

        optimizer_disc_A.zero_grad()

        # Loss on real images (should classify as real)
        real_loss_A = adversarial_criterion(discriminator_A(real_A), label_real)

        # Loss on fake images from replay buffer (should classify as fake)
        # Using replay buffer prevents mode collapse
        buffered_fake_A = buffer_fake_A.push_and_pop(fake_A)
        fake_loss_A = adversarial_criterion(
            discriminator_A(buffered_fake_A.detach()), label_fake
        )

        # Average discriminator loss
        total_disc_A_loss = (real_loss_A + fake_loss_A) / 2

        total_disc_A_loss.backward()
        optimizer_disc_A.step()

        # ====================================================================
        # TRAIN DISCRIMINATOR B (Domain B Authenticity Evaluator)
        # ====================================================================

        optimizer_disc_B.zero_grad()

        # Loss on real images
        real_loss_B = adversarial_criterion(discriminator_B(real_B), label_real)

        # Loss on fake images from replay buffer
        buffered_fake_B = buffer_fake_B.push_and_pop(fake_B)
        fake_loss_B = adversarial_criterion(
            discriminator_B(buffered_fake_B.detach()), label_fake
        )

        total_disc_B_loss = (real_loss_B + fake_loss_B) / 2

        total_disc_B_loss.backward()
        optimizer_disc_B.step()

        # Combined discriminator loss for logging
        avg_discriminator_loss = (total_disc_A_loss + total_disc_B_loss) / 2

        # ====================================================================
        # LOGGING AND VISUALIZATION
        # ====================================================================

        # Calculate total number of batches processed
        total_batches_processed = current_epoch * len(dataloader_train) + batch_idx

        # Print training progress to console
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f]"
            % (
                current_epoch,
                TOTAL_EPOCHS,
                batch_idx,
                len(dataloader_train),
                avg_discriminator_loss.item(),
                total_generator_loss.item(),
                combined_adversarial_loss.item(),
                combined_cycle_loss.item(),
                combined_identity_loss.item(),
            )
        )

        # Generate and save sample images at specified intervals
        if total_batches_processed % SAMPLE_GENERATION_FREQ == 0:
            generate_sample_images(total_batches_processed)

    # ========================================================================
    # END OF EPOCH OPERATIONS
    # ========================================================================

    # Update learning rates according to schedule
    scheduler_gen.step()
    scheduler_disc_A.step()
    scheduler_disc_B.step()

    # Save model checkpoints at specified intervals
    if CHECKPOINT_SAVE_FREQ != -1 and current_epoch % CHECKPOINT_SAVE_FREQ == 0:
        print(f"\nSaving checkpoints for epoch {current_epoch}...")

        torch.save(
            generator_A_to_B.state_dict(),
            "saved_models/%s/G_AB_%d.pth" % (DATASET_ID, current_epoch)
        )
        torch.save(
            generator_B_to_A.state_dict(),
            "saved_models/%s/G_BA_%d.pth" % (DATASET_ID, current_epoch)
        )
        torch.save(
            discriminator_A.state_dict(),
            "saved_models/%s/D_A_%d.pth" % (DATASET_ID, current_epoch)
        )
        torch.save(
            discriminator_B.state_dict(),
            "saved_models/%s/D_B_%d.pth" % (DATASET_ID, current_epoch)
        )