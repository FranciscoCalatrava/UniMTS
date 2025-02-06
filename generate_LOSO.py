import random
import yaml
from sklearn.model_selection import train_test_split


def generate_loso_distributions(dataset_subject_counts, val_split=0.2, seed=42):
    """
    Generates train, validation, and test splits for LOSO experiments.

    Parameters:
    - dataset_subject_counts (dict): Dictionary with dataset names as keys and subject count as values.
    - val_split (float): Percentage of remaining subjects to be used for validation (default 20%).
    - seed (int): Random seed for reproducibility.

    Returns:
    - dict: Nested dictionary containing LOSO splits for each dataset.
    """
    random.seed(seed)
    loso_splits = {}

    for dataset, num_subjects in dataset_subject_counts.items():
        loso_splits[dataset] = {}

        # Generate a list of subject IDs (assuming they are indexed from 1 to num_subjects)
        subjects = list(range(1, num_subjects + 1))

        for test_subject in subjects:
            remaining_subjects = [s for s in subjects if s != test_subject]

            # Split remaining subjects into train and validation
            train_subjects, val_subjects = train_test_split(remaining_subjects, test_size=val_split, random_state=seed)

            loso_splits[dataset][f"{test_subject}"] = {
                "train": train_subjects,
                "validation": val_subjects,
                "test": [test_subject]
            }

    return loso_splits


# Example usage:
dataset_subject_counts = {
    "USCHAD": 14,
    "UCIHAR": 30,
    "OPPORTUNITY": 4,
    "WISDM": 51,
    "DSADS": 8,
    "HARTH": 22,
    "WHARF": 17,  # Special handling may be required
    "MHEALTH": 10,
    "MHEALTH": 10,
    "UTDMHAD": 8,
    "MOTIONSENSE": 24,
    "WHAR": 22,
    "SHOAIB": 10,  # Might have issues with file handling
    "HAR70PLUS": 18,
    "MMACT": 20,  # Need to confirm split methodology
    "REALWORLD": 15,
    "TNDAHAR": 50,
    "UTCOMPLEX": 6,
    "REALDISP": 17,
    "UTDMHAD":8
}

# Generate LOSO splits
loso_data = generate_loso_distributions(dataset_subject_counts)

# Save to a YAML file
with open("LOSO_DISTRIBUTIONS.yaml", "w") as f:
    yaml.dump(loso_data, f, default_flow_style=False)

print("LOSO splits saved to loso_splits.yaml!")
