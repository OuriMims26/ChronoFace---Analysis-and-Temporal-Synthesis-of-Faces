import re
import shutil
import os
from glob import glob
import random

# Configuration: source folder containing raw images
SOURCE_FOLDER = "crop_part1"

# Configuration: number of images to reserve for testing per category
NUM_TEST_SAMPLES = 100

# Age thresholds for categorization
YOUNG_MIN_AGE = 15
YOUNG_MAX_AGE = 30
OLD_MIN_AGE = 50

# Retrieve all image paths from the source directory
image_paths = glob(f'{SOURCE_FOLDER}/*')

# Lists to store extracted metadata
ages = []
genders = []

# Extract age and gender information from filenames
# Expected filename format: "age_gender_*.jpg" (e.g., "25_1_12345.jpg")
for path in image_paths:
    # Use regex to capture age and gender from filename
    matches = re.findall(pattern=r'(\d+)_(\d)_', string=path)[0]
    extracted_age = int(matches[0])
    extracted_gender = int(matches[1])

    ages.append(extracted_age)
    genders.append(extracted_gender)

# Statistical analysis: count images in each age category
elderly_count = sum(1 for a in ages if a >= OLD_MIN_AGE)
youth_count = sum(1 for a in ages if YOUNG_MIN_AGE <= a <= YOUNG_MAX_AGE)

print(f"Dataset Statistics - Young faces: {youth_count}, Elderly faces: {elderly_count}")

# Create directory structure for training and testing
# Directory A: contains young faces
# Directory B: contains elderly faces
os.makedirs("train/A", exist_ok=True)
os.makedirs("train/B", exist_ok=True)
os.makedirs('test/A', exist_ok=True)
os.makedirs('test/B', exist_ok=True)

# Counters to track number of images copied to each category
young_counter = 0
elderly_counter = 0

# Organize images into train directories based on age criteria
for filename in glob(f"{SOURCE_FOLDER}/*"):
    # Parse age from filename
    age_match = re.findall(pattern=r"(\d+)_(\d)_", string=filename)[0]
    person_age = int(age_match[0])

    # Category A: Young individuals (15-30 years old)
    if YOUNG_MIN_AGE <= person_age <= YOUNG_MAX_AGE:
        destination = filename.replace(SOURCE_FOLDER, "train/A")
        shutil.copy(filename, destination)
        young_counter += 1

    # Category B: Elderly individuals (50+ years old)
    elif person_age >= OLD_MIN_AGE:
        destination = filename.replace(SOURCE_FOLDER, "train/B")
        shutil.copy(filename, destination)
        elderly_counter += 1

    # Skip images that don't fit either category (ages 31-49)
    else:
        continue

print(f"Training set created - Young: {young_counter} images, Elderly: {elderly_counter} images")

# Prepare test dataset by randomly sampling from training set
# This ensures model evaluation on unseen data

os.makedirs('test/A', exist_ok=True)
os.makedirs('test/B', exist_ok=True)

# Process both young (A) and elderly (B) categories
for category_path in ['train/A', 'train/B']:
    # Get all files in current category
    search_pattern = f'{category_path}/*'
    available_files = glob(search_pattern)

    # Randomly select samples for test set
    selected_samples = random.sample(available_files, k=NUM_TEST_SAMPLES)

    # Move selected files from train to test directory
    for sample_path in selected_samples:
        test_destination = sample_path.replace("train", "test")
        shutil.move(sample_path, test_destination)

print(f"Test set created - {NUM_TEST_SAMPLES} samples per category moved from train to test")