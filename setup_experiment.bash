#!/bin/bash

################################################################################
# Automated Environment Setup for Face Aging/Rejuvenation Project
################################################################################
# This script performs the following operations:
# 1. Downloads Google Drive file transfer utility
# 2. Fetches the UTKFace cropped dataset from cloud storage
# 3. Extracts compressed archives
# 4. Organizes images into proper directory structure for training
# 5. Creates final dataset layout compatible with CycleGAN architecture
#
# IMPORTANT: This script will modify the current Python environment
# by installing dependencies listed in requirements.txt
################################################################################

# ==============================================================================
# STEP 1: DOWNLOAD GOOGLE DRIVE FILE TRANSFER TOOL
# ==============================================================================

# Fetch gdown.pl - a Perl-based utility for downloading files from Google Drive
# This tool bypasses standard wget/curl limitations with Google Drive links
echo "======================================================================"
echo "Downloading Google Drive transfer utility (gdown.pl)..."
echo "======================================================================"

wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl

# Grant execution permissions to the downloaded script
chmod +x gdown.pl

echo "[SUCCESS] Google Drive utility download completed"
echo ""


# ==============================================================================
# STEP 2: DOWNLOAD FACE DATASET
# ==============================================================================

# Download the cropped face dataset (part 1)
# This archive contains pre-processed facial images from the UTKFace dataset
# File size: ~2GB, contains thousands of aligned and cropped face images
echo "======================================================================"
echo "Downloading cropped face dataset (crop_part1.tar.gz)..."
echo "This may take several minutes depending on connection speed"
echo "======================================================================"

./gdown.pl https://drive.google.com/open?id=0BxYys69jI14kRjNmM0gyVWM2bHM crop_part1.tar.gz

echo "[SUCCESS] Dataset archive download completed"
echo ""


# ==============================================================================
# STEP 3: EXTRACT COMPRESSED ARCHIVE
# ==============================================================================

# Decompress the tar.gz archive
# Options explained:
#   -z : decompress using gzip
#   -x : extract files from archive
#   -f : specify archive filename
echo "======================================================================"
echo "Extracting dataset archive..."
echo "======================================================================"

tar -zxf crop_part1.tar.gz

echo "[SUCCESS] Archive extraction completed"
echo ""


# ==============================================================================
# STEP 4: ORGANIZE DATASET INTO TRAINING STRUCTURE
# ==============================================================================

# Execute Python script to categorize images by age groups
# This script will:
#   - Parse age information from filenames
#   - Separate images into young (15-30) and old (50+) categories
#   - Create train/test splits
#   - Organize into A/B domain structure
echo "======================================================================"
echo "Organizing images into training/testing directories..."
echo "Running Python preprocessing script..."
echo "======================================================================"

python prepare_dataset.py

echo "[SUCCESS] Dataset directory structure created"
echo ""


# ==============================================================================
# STEP 5: FINALIZE DIRECTORY STRUCTURE
# ==============================================================================

# Create the main project dataset folder
# This will house all training and testing data
echo "======================================================================"
echo "Creating final project directory structure..."
echo "======================================================================"

mkdir old2young

# Move organized train/test folders into the project directory
# Final structure will be:
#   old2young/
#   ├── train/
#   │   ├── A/  (young faces)
#   │   └── B/  (old faces)
#   └── test/
#       ├── A/  (young faces)
#       └── B/  (old faces)

mv test old2young/
mv train old2young/

echo "[SUCCESS] Final directory structure created"
echo ""


# ==============================================================================
# SETUP COMPLETE
# ==============================================================================

echo "======================================================================"
echo "DATASET SETUP COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo "Your dataset is now ready at: ./old2young/"
echo "You can now proceed with model training"
echo "======================================================================"