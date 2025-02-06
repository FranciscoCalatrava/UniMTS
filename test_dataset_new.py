import sys
import os
from dataset.USCHAD import USCHAD
from dataset.OPPORTUNITY import OPPORTUNITY
from dataset.WHARF import WHARF
from dataset.UTDMHAD import UTDMHAD
from dataset.MOTIONSENSE import MOTIONSENSE
from dataset.WHAR import WHAR
from dataset.SHOAIB import SHOAIB
from dataset.HAR70PLUS import HAR70PLUS
from dataset.TNDAHAR import TNDAHAR
from dataset.DSADS import DSADS
from dataset.WISDM import WISDM
import numpy as np
import pandas as pd
import torch
import yaml
import h5py
import random
import itertools
import argparse
from scipy.spatial.transform import Rotation as R



def set_random_seed(seed):
    """
    Initialize all the random seeds to a specific number.

    Args:
    seed (int): The seed number to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f'Random seed set to {seed}')

def split_list_by_indices(original_list, size_of_part1):
    # Generate a list of indices from 0 to the length of the original list
    indices = list(range(len(original_list)))

    # Shuffle the indices to randomize them
    random.shuffle(indices)

    # Get indices for part1 and part2
    part1_indices = indices[:size_of_part1]
    part2_indices = indices[size_of_part1:]

    # Map these indices back to the original list
    part1 = [original_list[index] for index in part1_indices]
    part2 = [original_list[index] for index in part2_indices]

    return part1, part2

def get_dataset(dataset_name, experiment, current_dir, type = None):
    dataset_classes = {'USCHAD': USCHAD(train = [1,2,3,4,5,6,7,8,9], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'OPPORTUNITY': OPPORTUNITY(train = [1,3], validation = [2], test = [4], current_directory = current_dir),
                       'WHARF': WHARF(train = [1,2,3,4,5,6,7,8,9,16,17,15], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'UTDMHAD': UTDMHAD(train = [1,2,3,4,5], validation = [6,7], test = [8], current_directory = current_dir),
                       'MOTIONSENSE': MOTIONSENSE(train = [1,2,3,4,5,6,7,8,9,16,17,15,18,19,20,21,22,23,24], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'WHAR':WHAR(train = [1,2,3,4,5,6,7,8,9,16,17,15,18,19,20,21,22], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'SHOAIB': SHOAIB(train = [1,2,3,4,5,9,10], validation = [6,7], test = [8], current_directory = current_dir),
                       'HAR70PLUS': HAR70PLUS(train = [1,2,3,4,5,6,7,8,9,16,17,15,18], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'TNDAHAR': TNDAHAR(train = [1,2,3,4,5,6,7,8,9,16,17,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], validation = [10,11,12,13], test = [14], current_directory = current_dir),
                       'DSADS': DSADS(train = [1,2,3,4,5], validation = [6,7], test = [8], current_directory = current_dir),
                       'WISDM': WISDM(train = [1,2,3,4,5,6,7,8,9,16,17,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51], validation = [10,11,12,13], test = [14], current_directory = current_dir)}
    dataset_class = dataset_classes.get(dataset_name)

    if dataset_class:
        return dataset_class
    else:
        raise ValueError(f"No class defined for {dataset_name}")


def get_data(dataset_class,type):
    if dataset_class == None:
        raise ValueError(f"Empty dataset class")
    else:
        dataset_class.get_datasets()
        dataset_class.preprocessing()
        dataset_class.normalize()
        dataset_class.data_segmentation()
        dataset_class.prepare_dataset()

        train = [(a[0],a[1],a[2],1) for a in dataset_class.training_final]
        validation = [(a[0],a[1],a[2],1) for a in dataset_class.validation_final]
        test = [(a[0],a[1],a[2],1) for a in dataset_class.testing_final]

        # ##This way we ensure that the training set is the one that we want.
        # if type['train'] == 'rotated':
        #
        #     number_samples_train = int(len(train)*0.5)
        #     number_samples_validation = int(len(validation) * 0.5)
        #     ##We create the new_train
        #     new_train = augmented_dataset(train, number_samples_train,max_angle=15)
        #     new_validation = augmented_dataset(validation, number_samples_validation,max_angle=15)
        # else:
        #     new_train = train
        #     new_validation = validation
        return {"train": train, "validation": validation, "test": test}


def save_data(paths, dataset,type, args):
    if not os.path.exists(paths):
    # Create the directory
        os.makedirs(paths)
        print(f'Directory {paths} created')
    else:
        print(f'Directory {paths} already exists')
    for a in dataset.keys():
        if args.dataset == 'REALDISP':
            new_path = os.path.join(paths, a+f"_{type[a]}")
        else:
            new_path = os.path.join(paths, a)
        print(f"Saving {new_path}")
        # np.save(new_path, dataset[a], allow_pickle=True)
        # Save the list using h5py

        # print(dataset['train'][0][0])
        with h5py.File(new_path+'.h5', 'w') as hf:
            for i, (data, activity_label, person_label,domain) in enumerate(dataset[a]):
                grp = hf.create_group(f'item_{i}')
                grp.create_dataset('data', data=data)
                grp.create_dataset('activity_label', data=activity_label)
                grp.create_dataset('person_label', data=person_label)
                grp.create_dataset('domain_label', data=person_label)


def read_distribution(dataset, experiment_number):
    # Read the YAML file
    with open('./LOOCV_distribution.yaml', 'r') as file:
        data = yaml.safe_load(file)


    # Get the distribution for the given experiment number
    distribution = data[dataset]['distribution'].get(experiment_number, None)
    # print(distribution)



    if distribution is None:
        print(f"No distribution found for experiment number {experiment_number}")
    else:
        print(f"Distribution for experiment {experiment_number}: {distribution}")

    return {'train': distribution[0], 'validation': distribution[1], 'test': distribution[2]}



if __name__ == "__main__":


    print("\n\n")
    print("------------------------------------------")
    print("We initialize the seed")
    print("------------------------------------------")

    set_random_seed(0)

    dataset = 'WISDM'
    experiment = 1
    current_dir = './'


    data_prepared_dir = os.path.join(current_dir, f'datasets/{dataset}/prepared/{experiment}')

    paths = {'classification': os.path.join(data_prepared_dir,'classification/')}

    distribution = None

    print("\n\n")
    print("------------------------------------------")
    print("We construct the classification dataset")
    print("------------------------------------------")

    type = {'train': 'ideal',
            'validation':'ideal',
            'test':'ideal'}

    dataset_object = get_dataset(dataset, distribution, current_dir,type)
    classification_dataset = get_data(dataset_object,type)
    print()