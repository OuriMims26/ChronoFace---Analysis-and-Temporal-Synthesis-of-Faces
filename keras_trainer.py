from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from dataloader import DataLoader

import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os


# ============================================================================
# Implementation inspired by various GAN architectures for unpaired
# image-to-image translation tasks
# ============================================================================


class ImageDomainTranslator():
    """
    CycleGAN implementation for unpaired image-to-image translation.

    This class builds and trains a complete CycleGAN system consisting of:
    - Two generators for bidirectional translation (A↔B)
    - Two discriminators to evaluate image authenticity
    - Combined training with cycle consistency and identity preservation
    """

    def __init__(self):
        # ====================================================================
        # IMAGE SPECIFICATIONS
        # ====================================================================

        # Define input image dimensions
        self.image_height = 256
        self.image_width = 256
        self.num_channels = 3
        self.input_dimensions = (self.image_height, self.image_width, self.num_channels)

        # ====================================================================
        # DATA LOADING CONFIGURATION
        # ====================================================================

        # Dataset identifier for organizing outputs
        self.data_source = 'faceaging'

        # Initialize data loader with image resolution settings
        self.loader = DataLoader(
            dataset_name=self.data_source,
            img_res=(self.image_height, self.image_width)
        )

        # ====================================================================
        # DISCRIMINATOR OUTPUT CONFIGURATION
        # ====================================================================

        # Calculate PatchGAN discriminator output shape
        # PatchGAN evaluates NxN patches instead of entire image
        # This encourages high-frequency detail preservation
        patch_size = int(self.image_height / 2 ** 4)  # 4 downsampling layers
        self.discriminator_output_shape = (patch_size, patch_size, 1)

        # ====================================================================
        # NETWORK ARCHITECTURE PARAMETERS
        # ====================================================================

        # Base number of filters for generator network
        self.generator_base_filters = 32

        # Base number of filters for discriminator network
        self.discriminator_base_filters = 64

        # ====================================================================
        # LOSS FUNCTION WEIGHTS
        # ====================================================================

        # Cycle consistency loss weight (ensures A→B→A ≈ A)
        # Higher values enforce stronger reconstruction constraints
        self.cycle_loss_coefficient = 10.0

        # Identity mapping loss weight (preserves color/style when unnecessary)
        # Set to 10% of cycle loss as recommended in original paper
        self.identity_loss_coefficient = 0.1 * self.cycle_loss_coefficient

        # ====================================================================
        # OPTIMIZER SETUP
        # ====================================================================

        # Adam optimizer with parameters from original CycleGAN paper
        # Learning rate: 0.0002, Beta1: 0.5 (slower momentum for stability)
        optimization_algorithm = Adam(0.0002, 0.5)

        # ====================================================================
        # BUILD AND COMPILE DISCRIMINATORS
        # ====================================================================

        # Discriminator for domain A (evaluates if image belongs to domain A)
        self.discriminator_domain_A = self.construct_discriminator()

        # Discriminator for domain B (evaluates if image belongs to domain B)
        self.discriminator_domain_B = self.construct_discriminator()

        # Compile discriminators with MSE loss (LSGAN variant)
        # MSE provides smoother gradients than binary cross-entropy
        self.discriminator_domain_A.compile(
            loss='mse',
            optimizer=optimization_algorithm,
            metrics=['accuracy']
        )

        self.discriminator_domain_B.compile(
            loss='mse',
            optimizer=optimization_algorithm,
            metrics=['accuracy']
        )

        # ====================================================================
        # BUILD GENERATOR COMPUTATIONAL GRAPH
        # ====================================================================

        # Generator A→B: transforms images from domain A to domain B
        self.generator_A_to_B = self.construct_generator()

        # Generator B→A: transforms images from domain B to domain A
        self.generator_B_to_A = self.construct_generator()

        # Define input placeholders for both domains
        input_domain_A = Input(shape=self.input_dimensions)
        input_domain_B = Input(shape=self.input_dimensions)

        # Forward cycle: A → B (translation)
        translated_to_B = self.generator_A_to_B(input_domain_A)

        # Forward cycle: B → A (translation)
        translated_to_A = self.generator_B_to_A(input_domain_B)

        # Backward cycle: A → B → A (reconstruction)
        # This enforces cycle consistency: reconstructed image should match original
        reconstructed_A = self.generator_B_to_A(translated_to_B)

        # Backward cycle: B → A → B (reconstruction)
        reconstructed_B = self.generator_A_to_B(translated_to_A)

        # Identity mapping: domain A through B→A generator
        # If input is already in domain A, generator should leave it unchanged
        identity_mapped_A = self.generator_B_to_A(input_domain_A)

        # Identity mapping: domain B through A→B generator
        identity_mapped_B = self.generator_A_to_B(input_domain_B)

        # Freeze discriminators during generator training
        # This prevents discriminators from being updated during generator phase
        self.discriminator_domain_A.trainable = False
        self.discriminator_domain_B.trainable = False

        # Evaluate translated images with discriminators
        # These outputs measure how "real" the generated images appear
        authenticity_score_A = self.discriminator_domain_A(translated_to_A)
        authenticity_score_B = self.discriminator_domain_B(translated_to_B)

        # ====================================================================
        # COMBINED MODEL FOR GENERATOR TRAINING
        # ====================================================================

        # Create unified model that trains both generators simultaneously
        # This model takes real images from both domains and outputs:
        # 1-2: Discriminator evaluations (adversarial loss)
        # 3-4: Reconstructed images (cycle consistency loss)
        # 5-6: Identity mapped images (identity loss)
        self.unified_model = Model(
            inputs=[input_domain_A, input_domain_B],
            outputs=[
                authenticity_score_A, authenticity_score_B,
                reconstructed_A, reconstructed_B,
                identity_mapped_A, identity_mapped_B
            ]
        )

        # Compile with multiple loss functions
        # MSE for adversarial (smoother than BCE)
        # MAE for reconstruction (more robust to outliers than MSE)
        self.unified_model.compile(
            loss=[
                'mse', 'mse',  # Adversarial losses
                'mae', 'mae',  # Cycle consistency losses
                'mae', 'mae'  # Identity losses
            ],
            loss_weights=[
                1, 1,  # Equal weight for adversarial losses
                self.cycle_loss_coefficient, self.cycle_loss_coefficient,
                self.identity_loss_coefficient, self.identity_loss_coefficient
            ],
            optimizer=optimization_algorithm
        )

    def construct_generator(self):
        """
        Build U-Net style generator architecture.

        U-Net uses skip connections to preserve spatial information
        during the encoding-decoding process, which is crucial for
        maintaining fine details in image translation tasks.

        Returns:
            Model: Keras functional model (input_image → translated_image)
        """

        def encoding_block(input_layer, num_filters, kernel_size=4):
            """
            Downsampling block for encoder path.

            Each block reduces spatial dimensions by half while
            increasing feature depth for abstract representations.

            Args:
                input_layer: Previous layer output
                num_filters: Number of convolutional filters
                kernel_size: Convolution kernel size

            Returns:
                Encoded feature map with reduced spatial dimensions
            """
            # Strided convolution for downsampling (more learnable than pooling)
            encoded = Conv2D(num_filters, kernel_size=kernel_size,
                             strides=2, padding='same')(input_layer)

            # LeakyReLU prevents dying ReLU problem
            encoded = LeakyReLU(alpha=0.2)(encoded)

            # Instance normalization (better than batch norm for style transfer)
            # Normalizes each instance independently
            encoded = InstanceNormalization()(encoded)

            return encoded

        def decoding_block(input_layer, skip_connection, num_filters,
                           kernel_size=4, dropout_rate=0):
            """
            Upsampling block for decoder path with skip connections.

            Skip connections from encoder help recover spatial details
            lost during downsampling, improving reconstruction quality.

            Args:
                input_layer: Previous layer output
                skip_connection: Corresponding encoder layer for concatenation
                num_filters: Number of convolutional filters
                kernel_size: Convolution kernel size
                dropout_rate: Dropout probability (0 = no dropout)

            Returns:
                Decoded feature map with increased spatial dimensions
            """
            # Nearest-neighbor upsampling (doubles spatial dimensions)
            decoded = UpSampling2D(size=2)(input_layer)

            # Convolution to refine upsampled features
            decoded = Conv2D(num_filters, kernel_size=kernel_size,
                             strides=1, padding='same', activation='relu')(decoded)

            # Optional dropout for regularization
            if dropout_rate:
                decoded = Dropout(dropout_rate)(decoded)

            # Instance normalization for stable training
            decoded = InstanceNormalization()(decoded)

            # Concatenate with skip connection from encoder
            # This preserves fine-grained details from earlier layers
            decoded = Concatenate()([decoded, skip_connection])

            return decoded

        # ====================================================================
        # ENCODER PATH (Downsampling)
        # ====================================================================

        # Input layer
        network_input = Input(shape=self.input_dimensions)

        # Progressive downsampling with increasing feature depth
        # Each layer captures increasingly abstract features
        encoder_1 = encoding_block(network_input, self.generator_base_filters)
        encoder_2 = encoding_block(encoder_1, self.generator_base_filters * 2)
        encoder_3 = encoding_block(encoder_2, self.generator_base_filters * 4)
        encoder_4 = encoding_block(encoder_3, self.generator_base_filters * 8)

        # ====================================================================
        # DECODER PATH (Upsampling with Skip Connections)
        # ====================================================================

        # Progressive upsampling with decreasing feature depth
        # Skip connections preserve spatial information
        decoder_1 = decoding_block(encoder_4, encoder_3, self.generator_base_filters * 4)
        decoder_2 = decoding_block(decoder_1, encoder_2, self.generator_base_filters * 2)
        decoder_3 = decoding_block(decoder_2, encoder_1, self.generator_base_filters)

        # Final upsampling to match input resolution
        decoder_4 = UpSampling2D(size=2)(decoder_3)

        # Output layer: map to RGB image with tanh activation
        # Tanh outputs values in [-1, 1] range (standard for GANs)
        final_output = Conv2D(
            self.num_channels,
            kernel_size=4,
            strides=1,
            padding='same',
            activation='tanh'
        )(decoder_4)

        return Model(network_input, final_output)

    def construct_discriminator(self):
        """
        Build PatchGAN discriminator architecture.

        PatchGAN classifies overlapping image patches as real/fake
        rather than classifying the entire image. This encourages
        the generator to produce realistic high-frequency details.

        Returns:
            Model: Keras functional model (input_image → validity_map)
        """

        def discriminator_layer(input_layer, num_filters, kernel_size=4,
                                apply_normalization=True):
            """
            Convolutional block for discriminator.

            Args:
                input_layer: Previous layer output
                num_filters: Number of convolutional filters
                kernel_size: Convolution kernel size
                apply_normalization: Whether to apply instance normalization

            Returns:
                Discriminator feature map
            """
            # Strided convolution for downsampling
            features = Conv2D(num_filters, kernel_size=kernel_size,
                              strides=2, padding='same')(input_layer)

            # LeakyReLU activation for better gradient flow
            features = LeakyReLU(alpha=0.2)(features)

            # Instance normalization (skipped in first layer as per convention)
            if apply_normalization:
                features = InstanceNormalization()(features)

            return features

        # Input layer
        discriminator_input = Input(shape=self.input_dimensions)

        # Progressive downsampling with increasing receptive field
        # First layer: no normalization (standard practice)
        layer_1 = discriminator_layer(
            discriminator_input,
            self.discriminator_base_filters,
            apply_normalization=False
        )

        # Subsequent layers: double filters and apply normalization
        layer_2 = discriminator_layer(layer_1, self.discriminator_base_filters * 2)
        layer_3 = discriminator_layer(layer_2, self.discriminator_base_filters * 4)
        layer_4 = discriminator_layer(layer_3, self.discriminator_base_filters * 8)

        # Output layer: single channel validity map
        # Each pixel in this map represents real/fake classification for a patch
        validity_output = Conv2D(1, kernel_size=4, strides=1, padding='same')(layer_4)

        return Model(discriminator_input, validity_output)

    def train(self, epochs, batch_size=1, sample_interval=50):
        """
        Execute the complete training procedure.

        Training alternates between:
        1. Discriminator updates (distinguish real from fake)
        2. Generator updates (fool discriminators + maintain consistency)

        Args:
            epochs: Number of complete dataset passes
            batch_size: Number of images per training batch
            sample_interval: Frequency of sample image generation (in batches)
        """

        # Record training start time for elapsed time tracking
        training_start_time = datetime.datetime.now()

        # Prepare ground truth labels for adversarial training
        # real_label: used for real images (all ones)
        # fake_label: used for generated images (all zeros)
        real_label = np.ones((batch_size,) + self.discriminator_output_shape)
        fake_label = np.zeros((batch_size,) + self.discriminator_output_shape)

        # ====================================================================
        # MAIN TRAINING LOOP
        # ====================================================================

        for current_epoch in range(epochs):

            # Iterate through all batches in dataset
            for batch_index, (batch_A, batch_B) in enumerate(
                    self.loader.load_batch(batch_size)
            ):

                # ============================================================
                # PHASE 1: TRAIN DISCRIMINATORS
                # ============================================================

                # Generate fake images by translating real images
                generated_B = self.generator_A_to_B.predict(batch_A)
                generated_A = self.generator_B_to_A.predict(batch_B)

                # --- Train Discriminator A ---
                # Learn to distinguish real domain A images from fake ones

                # Loss on real images (should output 1)
                loss_real_A = self.discriminator_domain_A.train_on_batch(
                    batch_A, real_label
                )

                # Loss on fake images (should output 0)
                loss_fake_A = self.discriminator_domain_A.train_on_batch(
                    generated_A, fake_label
                )

                # Average loss for domain A discriminator
                combined_loss_A = 0.5 * np.add(loss_real_A, loss_fake_A)

                # --- Train Discriminator B ---
                # Learn to distinguish real domain B images from fake ones

                loss_real_B = self.discriminator_domain_B.train_on_batch(
                    batch_B, real_label
                )

                loss_fake_B = self.discriminator_domain_B.train_on_batch(
                    generated_B, fake_label
                )

                combined_loss_B = 0.5 * np.add(loss_real_B, loss_fake_B)

                # Total discriminator loss (average of both domains)
                total_discriminator_loss = 0.5 * np.add(
                    combined_loss_A, combined_loss_B
                )

                # ============================================================
                # PHASE 2: TRAIN GENERATORS
                # ============================================================

                # Train generators through combined model
                # Target outputs:
                # - Translated images should fool discriminators (real_label)
                # - Reconstructed images should match originals (batch_A/B)
                # - Identity mappings should match inputs (batch_A/B)
                generator_loss = self.unified_model.train_on_batch(
                    [batch_A, batch_B],
                    [
                        real_label, real_label,  # Fool discriminators
                        batch_A, batch_B,  # Cycle consistency
                        batch_A, batch_B  # Identity preservation
                    ]
                )

                # ============================================================
                # LOGGING AND VISUALIZATION
                # ============================================================

                # Calculate elapsed training time
                time_elapsed = datetime.datetime.now() - training_start_time

                # Print comprehensive training statistics
                # Loss breakdown:
                # - D loss: discriminator performance
                # - G loss: total generator loss
                # - adv: adversarial component
                # - recon: reconstruction/cycle component
                # - id: identity preservation component
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] "
                    "[G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s"
                    % (
                        current_epoch, epochs,
                        batch_index, self.loader.n_batches,
                        total_discriminator_loss[0],
                        100 * total_discriminator_loss[1],
                        generator_loss[0],
                        np.mean(generator_loss[1:3]),  # Adversarial losses
                        np.mean(generator_loss[3:5]),  # Cycle losses
                        np.mean(generator_loss[5:6]),  # Identity loss
                        time_elapsed
                    )
                )

                # Generate and save sample images at regular intervals
                if batch_index % sample_interval == 0:
                    self.generate_samples(current_epoch, batch_index)

    def generate_samples(self, epoch_num, batch_num):
        """
        Generate and save visualization of translation results.

        Creates a 2x3 grid showing:
        - Row 1: Domain A → B → A (original, translated, reconstructed)
        - Row 2: Domain B → A → B (original, translated, reconstructed)

        Args:
            epoch_num: Current epoch number (for filename)
            batch_num: Current batch number (for filename)
        """

        # Create output directory if it doesn't exist
        try:
            os.makedirs('images/%s' % self.data_source)
        except:
            pass

        # Define visualization grid dimensions
        num_rows, num_cols = 2, 3

        # Load test samples from both domains
        test_batch_A = self.loader.load_data(
            domain="A", batch_size=1, is_testing=True
        )
        test_batch_B = self.loader.load_data(
            domain="B", batch_size=1, is_testing=True
        )

        # ====================================================================
        # GENERATE TRANSLATIONS AND RECONSTRUCTIONS
        # ====================================================================

        # Translate domain A to domain B
        translated_B = self.generator_A_to_B.predict(test_batch_A)

        # Translate domain B to domain A
        translated_A = self.generator_B_to_A.predict(test_batch_B)

        # Reconstruct original images (cycle consistency verification)
        reconstructed_A = self.generator_B_to_A.predict(translated_B)
        reconstructed_B = self.generator_A_to_B.predict(translated_A)

        # Combine all images for visualization
        # Order: [original_A, translated_B, reconstructed_A,
        #         original_B, translated_A, reconstructed_B]
        visualization_grid = np.concatenate([
            test_batch_A, translated_B, reconstructed_A,
            test_batch_B, translated_A, reconstructed_B
        ])

        # Rescale from [-1, 1] to [0, 1] for display
        visualization_grid = 0.5 * visualization_grid + 0.5

        # ====================================================================
        # CREATE AND SAVE VISUALIZATION
        # ====================================================================

        # Column titles for the visualization
        column_headers = ['Original', 'Translated', 'Reconstructed']

        # Create matplotlib figure
        figure, axes = plt.subplots(num_rows, num_cols)

        image_counter = 0
        for row in range(num_rows):
            for col in range(num_cols):
                # Display image
                axes[row, col].imshow(visualization_grid[image_counter])

                # Set column title
                axes[row, col].set_title(column_headers[col])

                # Remove axis ticks for cleaner appearance
                axes[row, col].axis('off')

                image_counter += 1

        # Save figure to disk
        output_filename = "images/%d_%d.png" % (epoch_num, batch_num)
        figure.savefig(output_filename)
        plt.close()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Initialize CycleGAN model
    translation_model = ImageDomainTranslator()

    # Begin training process
    # 500 epochs with visualization every 200 batches
    translation_model.train(epochs=500, batch_size=1, sample_interval=200)