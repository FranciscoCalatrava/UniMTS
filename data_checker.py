import torch
from torch.utils.data import DataLoader, TensorDataset
import pkgutil
import importlib
import os
import yaml
import math
import traceback



def get_correct_parameters_dataset(name):
    """
    Returns the expected parameters for a given dataset.
    """
    dataset_parameters = {
        "OPPORTUNITY": {
            "expected_signal_shape": (97, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 4,
            "num_subject": 4
        },
        "UCIHAR": {
            "expected_signal_shape": (9, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 6,
            "num_subject": 30
        },
        "MOTIONSENSE": {
            "expected_signal_shape": (6, 256),  # (num_channels, window_size)
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),  # Normalized sensor data
            "num_classes": 6,
            "num_subject": 24
        },
        "WHAR": {
            "expected_signal_shape": (6, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 7,
            "num_subject": 22
        },
        "SHOAIB": {
            "expected_signal_shape": (30, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 7,
            "num_subject": 10
        },
        "HAR70PLUS": {
            "expected_signal_shape": (6, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 7,
            "num_subject": 18
        },
        "REALWORLD": {
            "expected_signal_shape": None,
            "expected_dtype": None,
            "expected_range": None,
            "num_classes": None,
            "num_subject": None
        },
        "TNDAHAR": {
            "expected_signal_shape": (30, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 8,
            "num_subject": 50
        },
        "PAMAP2": {
            "expected_signal_shape": (18, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 12,
            "num_subject": 8
        },
        "USCHAD": {
            "expected_signal_shape": (6, 256),  # (num_channels, window_size)
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),  # Normalized sensor data
            "num_classes": 12,
            "num_subject": 14
        },
        "MHEALTH": {
            "expected_signal_shape": (15, 256),  # (num_channels, window_size)
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),  # Normalized sensor data
            "num_classes": 12,
            "num_subject": 10
        },
        "HARTH": {
            "expected_signal_shape": (6, 256),  # (num_channels, window_size)
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),  # Normalized sensor data
            "num_classes": 12,
            "num_subject": 22
        },
        "WHARF": {
            "expected_signal_shape": (3, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 14,
            "num_subject": 17
        },
        "WISDM": {
            "expected_signal_shape": None,
            "expected_dtype": None,
            "expected_range": None,
            "num_classes": 18,
            "num_subject": 51
        },
        "DSADS": {
            "expected_signal_shape": (30, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 19,
            "num_subject": 8
        },
        "UTDMHAD": {
            "expected_signal_shape": (6, 256),
            "expected_dtype": torch.float32,
            "expected_range": (-1.0, 1.0),
            "num_classes": 27,
            "num_subject": 8
        },

    }

    if name not in dataset_parameters:
        print(f"[WARNING] Dataset '{name}' not found in predefined parameters.")
        return {
            "expected_signal_shape": None,
            "expected_dtype": None,
            "expected_range": None,
            "num_classes": None,
            "num_subject": None,
        }
    return dataset_parameters[name]


def import_dataset_classes(package_name):
    """
    Dynamically import all modules from the specified package and return a dictionary mapping
    the module name (or a derived key) to the dataset class found in that module.
    """
    dataset_classes = {}
    # Import the package itself
    package = importlib.import_module(package_name)

    # Iterate over all modules in the package directory
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            continue  # Skip subpackages

        full_module_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        # Assume the dataset class name is the same as the module name.
        dataset_class = getattr(module, module_name, None)

        # If not found, try capitalizing the module name
        if dataset_class is None:
            dataset_class = getattr(module, module_name.capitalize(), None)

        if dataset_class is not None:
            dataset_classes[module_name] = dataset_class
            print(f"Imported dataset class '{dataset_class.__name__}' from module '{full_module_name}'")
        else:
            print(f"[Warning] Could not find a dataset class in module '{full_module_name}'")

    return dataset_classes


def check_tensor(tensor, name, expected_shape=None, expected_dtype=None, expected_range=None):
    """Check basic properties of a tensor."""
    passed = True

    # Check that the input is a torch.Tensor
    if not isinstance(tensor, torch.Tensor):
        print(f"[ERROR] {name} is not a torch.Tensor!")
        return False

    # Check shape (ignoring the batch dimension)
    if expected_shape is not None:
        if tensor.ndim < 2:
            print(f"[ERROR] {name} should have at least 2 dimensions (batch + data shape), but got {tensor.ndim}.")
            passed = False
        elif tensor.shape[1:] != tuple(expected_shape):
            print(f"[ERROR] {name} shape {tensor.shape[1:]} does not match expected {expected_shape}.")
            passed = False
        else:
            print(f"[OK] {name} shape check passed: {tensor.shape[1:]}")

    # Check data type
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            print(f"[ERROR] {name} dtype {tensor.dtype} does not match expected {expected_dtype}.")
            passed = False
        else:
            print(f"[OK] {name} dtype check passed: {tensor.dtype}")

    # Check value range
    if expected_range is not None:
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        if tensor_min < expected_range[0] or tensor_max > expected_range[1]:
            print(f"[ERROR] {name} values are out of expected range {expected_range} (min: {tensor_min}, max: {tensor_max}).")
            passed = False
        else:
            print(f"[OK] {name} value range check passed: ({tensor_min}, {tensor_max})")

    # Check for NaN and Inf values
    if torch.isnan(tensor).any():
        print(f"[ERROR] {name} contains NaN values!")
        passed = False
    if torch.isinf(tensor).any():
        print(f"[ERROR] {name} contains Inf values!")
        passed = False

    return passed


def test_dataloader(loader, dataset_params):
    """Retrieve one batch from the loader and run tests on inertial signals and labels."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("[ERROR] DataLoader is empty!")
        return False

    # Assume each batch is a tuple (signals, labels)
    signals, labels = batch

    print(f"\nTesting dataset: {dataset_params.get('name', 'Unnamed Dataset')}")
    passed_signals = check_tensor(
        signals,
        "Signals",
        expected_shape=dataset_params.get("expected_signal_shape"),
        expected_dtype=dataset_params.get("expected_dtype"),
        expected_range=dataset_params.get("expected_range")
    )

    # Check labels (only dtype and valid range if num_classes is provided)
    passed_labels = check_tensor(
        labels,
        "Labels",
        expected_dtype=dataset_params.get("label_dtype", torch.long)
    )

    # Optionally, check that label values are within the correct range
    num_classes = dataset_params.get("num_classes")
    if num_classes is not None:
        label_min = labels.min().item()
        label_max = labels.max().item()
        if label_min < 0 or label_max >= num_classes:
            print(f"[ERROR] Labels have values out of expected range 0 to {num_classes - 1} (min: {label_min}, max: {label_max}).")
            passed_labels = False
        else:
            print(f"[OK] Label values in expected range: ({label_min}, {label_max})")

    return passed_signals and passed_labels


def test_subjects(num_subject_train, num_subject_val, num_subject_test, general):
    """
    Checks the subject split across partitions.
      - Test partition must always have exactly 1 subject.
      - Validation partition must have 20% (rounded up) of the train partition's subjects.
    """
    passed = True
    expected_test = 1
    if num_subject_test != expected_test:
        print(f"[ERROR] Test partition has {num_subject_test} subjects; expected {expected_test}.")
        passed = False
    else:
        print(f"[OK] Test partition subject count is {num_subject_test} (expected {expected_test}).")

    print(f"General is {general}")

    # Calculate expected number of subjects in the validation partition
    expected_val = math.ceil((general-1) * 0.2)
    if num_subject_val != expected_val:
        print(f"[ERROR] Validation partition has {num_subject_val} subjects; expected {expected_val} (20% of {num_subject_train} train subjects).")
        passed = False
    else:
        print(f"[OK] Validation partition subject count is {num_subject_val} (expected {expected_val}).")

    return passed


def get_data(dataset_instance, data_type):
    """
    Calls the necessary methods on the dataset instance to generate training, validation, and test splits.
    """
    if dataset_instance is None:
        raise ValueError("Empty dataset instance")
    else:
        dataset_instance.get_datasets()
        dataset_instance.preprocessing()
        dataset_instance.normalize()
        dataset_instance.data_segmentation()
        dataset_instance.prepare_dataset()

        # Expecting that these attributes are set by prepare_dataset()
        train = [(a[0], a[1], a[2]) for a in dataset_instance.training_final]
        validation = [(a[0], a[1], a[2]) for a in dataset_instance.validation_final]
        test = [(a[0], a[1], a[2]) for a in dataset_instance.testing_final]
        print(f"The lenght of the training data is {len(train)}")
        print(f"The lenght of the validation data is {len(validation)}")
        print(f"The lenght of the testing data is {len(test)}")

    return {"train": train, "validation": validation, "test": test}


def sequential_dataset_check(datasets, batch_size=32):
    """
    Iterates through each dataset instance, converts its LOSO split into a DataLoader,
    and performs sanity checks including the number of subjects per partition.
    """
    # Load LOSO distributions (if needed for future use)
    with open("LOSO_DISTRIBUTIONS.yaml", "r") as f:
        loso_distribution = yaml.safe_load(f)

    results = {}

    for dataset_name, dataset_instance in datasets.items():
        print("=" * 80)
        print(f"Checking dataset: {dataset_name}")

        try:
            # Load the dataset splits using the provided dataset instance.
            data_splits = get_data(dataset_instance, data_type={"train": "default"})

            # Define a helper to convert list data into a DataLoader.
            def convert_to_dataloader(data):
                inputs = torch.tensor([x[0] for x in data], dtype=torch.float32)  # Sensor signals
                labels = torch.tensor([x[1] for x in data], dtype=torch.long)     # Activity labels
                ds = TensorDataset(inputs, labels)
                return DataLoader(ds, batch_size=batch_size, shuffle=True)

            train_loader = convert_to_dataloader(data_splits["train"])
            val_loader = convert_to_dataloader(data_splits["validation"])
            test_loader = convert_to_dataloader(data_splits["test"])

            # Compute the expected signal shape from the first training sample (if available)
            if len(data_splits["train"]) > 0:
                sample_signal = data_splits["train"][0][0]
                expected_signal_shape = tuple(torch.tensor(sample_signal).shape)
            else:
                expected_signal_shape = None

            # Compute number of classes from the labels (second element in each sample).
            num_classes_train = len(set([x[1] for x in data_splits["train"]])) if data_splits["train"] else None
            num_classes_val = len(set([x[1] for x in data_splits["validation"]])) if data_splits["validation"] else None
            num_classes_test = len(set([x[1] for x in data_splits["test"]])) if data_splits["test"] else None

            # Compute number of subjects from the third element in each sample.
            num_subject_train = len(set([x[2] for x in data_splits["train"]])) if data_splits["train"] else None
            num_subject_val = len(set([x[2] for x in data_splits["validation"]])) if data_splits["validation"] else None
            num_subject_test = len(set([x[2] for x in data_splits["test"]])) if data_splits["test"] else None

            correct_parameters = get_correct_parameters_dataset(dataset_name)
            expected_signal_shape = correct_parameters["expected_signal_shape"]
            num_classes_train = correct_parameters["num_classes"]
            num_classes_val = correct_parameters["num_classes"]
            num_classes_test = correct_parameters["num_classes"]
            num_subject_train =  math.ceil((correct_parameters["num_subject"] - 1) * 0.8)
            num_subject_val = math.ceil((correct_parameters["num_subject"] - 1) * 0.2)
            num_subject_test = 1

            # Create dataset parameter dictionaries.
            ds_params_train = {
                "name": dataset_name,
                "expected_signal_shape": expected_signal_shape,
                "expected_dtype": torch.float32,
                "num_classes": num_classes_train,
                "num_subjects": num_subject_train
            }
            ds_params_val = {
                "name": dataset_name,
                "expected_signal_shape": expected_signal_shape,
                "expected_dtype": torch.float32,
                "num_classes": num_classes_val,
                "num_subjects": num_subject_val
            }
            ds_params_test = {
                "name": dataset_name,
                "expected_signal_shape": expected_signal_shape,
                "expected_dtype": torch.float32,
                "num_classes": num_classes_test,
                "num_subjects": num_subject_test
            }

            # Run sanity checks on each DataLoader.
            train_passed = test_dataloader(train_loader, ds_params_train)
            val_passed = test_dataloader(val_loader, ds_params_val)
            test_passed = test_dataloader(test_loader, ds_params_test)

            # Test the subject splits.
            subject_passed = test_subjects(num_subject_train, num_subject_val, num_subject_test, (num_subject_train + num_subject_val + num_subject_test))

            final_passed = train_passed and val_passed and test_passed and subject_passed

            results[dataset_name] = {
                "train_passed": train_passed,
                "validation_passed": val_passed,
                "test_passed": test_passed,
                "subject_passed": subject_passed,
                "final": final_passed
            }

            print(f"[RESULT] {dataset_name} - Train: {train_passed}, Validation: {val_passed}, Test: {test_passed}, Subjects: {subject_passed}")

        except Exception as e:
            print(f"[ERROR] Failed to check dataset {dataset_name}: {str(e)}")
            traceback.print_exc()
            results[dataset_name] = {"error": str(e)}

    print("=" * 80)
    print("\nFinal Summary:")
    for ds_name, status in results.items():
        print(f"{ds_name}: {status}")

    return results


def main():
    # Import dataset classes from the 'dataset' package
    datasets_classes = import_dataset_classes("dataset")

    dataset_dict = {}

    # For each dataset class, instantiate it using the LOSO distributions.
    with open("LOSO_DISTRIBUTIONS.yaml", "r") as f:
        distribution = yaml.safe_load(f)

    for name, ds_class in datasets_classes.items():
        print(f"Instantiating dataset: {name} -> {ds_class}")
        # It is assumed that the YAML file has a structure like:
        # { <dataset_name>: { '1': { 'train': ..., 'validation': ..., 'test': ... } } }
        dataset_experiment_distribution = distribution.get(name, {}).get('1', {})
        if not dataset_experiment_distribution:
            print(f"[Warning] No LOSO distribution found for dataset {name}. Skipping...")
            continue
        # Create an instance of the dataset.
        dataset_dict[name] = ds_class(
            train=dataset_experiment_distribution.get('train'),
            validation=dataset_experiment_distribution.get('validation'),
            test=dataset_experiment_distribution.get('test'),
            current_directory='./'
        )

    print("\nDataset instances created:")
    print(dataset_dict)

    # Run the sequential dataset checks
    sequential_dataset_check(dataset_dict)


if __name__ == "__main__":
    main()
