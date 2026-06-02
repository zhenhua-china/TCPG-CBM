# https://github.com/mateoespinosa/cem/blob/main/cem/data/celeba_loader.py

import numpy as np
import os
import torch
import torchvision
import logging

from pathlib import Path
from torchvision import transforms
from pytorch_lightning import seed_everything

###############################################################################
## GLOBAL VARIABLES
###############################################################################


# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR



#########################################################
## CONCEPT INFORMATION REGARDING CelebA
#########################################################

CELEBA_CONFIG = dict(
    image_size=64,
    num_classes=256,
    use_imbalance=True,
    use_binary_vector_class=True,
    num_concepts=6,
    label_binary_width=1,
    label_dataset_subsample=12,
    num_hidden_concepts=2,
    selected_concepts=False,
)




SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################


def generate_data(root_dir, resol, batch_size, num_workers, config=CELEBA_CONFIG, seed=0, output_dataset_vars=False):
    seed_everything(42) # set as 42 for all circumstances
    concept_group_map = None
    use_binary_vector_class = config.get('use_binary_vector_class', False)
    concept_names = None
    concept_indices = None
    if use_binary_vector_class:
        # Now reload by transform the labels accordingly
        width = config.get('label_binary_width', 5)
        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(
                    str(int(np.sum(concepts[i : i + width]) > 0))
                )
            return int("".join(binary_repr), 2)

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['attr'],
        )

        concept_freq = np.sum(
            celeba_train_data.attr.cpu().detach().numpy(),
            axis=0
        ) / celeba_train_data.attr.shape[0]
        # logging.debug(f"Concept frequency is: {concept_freq}")
        sorted_concepts = list(map(
            lambda x: x[0],
            sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
        ))
        num_concepts = config.get(
            'num_concepts',
            celeba_train_data.attr.shape[-1],
        )
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        if config.get('num_hidden_concepts', 0):
            num_hidden = config.get('num_hidden_concepts', 0)
            hidden_concepts = sorted(
                sorted_concepts[
                    num_concepts:min(
                        (num_concepts + num_hidden),
                        len(sorted_concepts)
                    )
                ]
            )
        else:
            hidden_concepts = []
        concept_indices = concept_idxs + hidden_concepts
        concept_names = [CONCEPT_SEMANTICS[i] for i in concept_idxs]
        # logging.debug(f"Selecting concepts: {concept_idxs}")
        # logging.debug(f"\tAnd hidden concepts: {hidden_concepts}")
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            transform=transforms.Compose([
                transforms.Resize(resol),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_indices),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(map(
                lambda x: _binarize(
                    x.cpu().detach().numpy(),
                    selected=(concept_indices),
                    width=width,
                ),
                celeba_train_data.attr
            )),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        num_classes = len(label_remap)
        train_transform = transforms.Compose([
            transforms.Resize(resol),
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(resol),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ])
        celeba_all_aug = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            transform=train_transform,
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_indices),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        celeba_all_eval = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            transform=eval_transform,
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_indices),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        factor = config.get('label_dataset_subsample', 1)
        total_len = len(celeba_all_eval)
        if factor != 1:
            idxs = np.random.choice(np.arange(0, total_len), replace=False, size=total_len // factor)
        else:
            idxs = np.arange(0, total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(idxs)
        train_samples = int(0.7 * len(idxs))
        test_samples = int(0.2 * len(idxs))
        val_samples = len(idxs) - test_samples - train_samples
        train_indices = idxs[:train_samples]
        test_indices = idxs[train_samples:train_samples + test_samples]
        val_indices = idxs[train_samples + test_samples:train_samples + test_samples + val_samples]
        celeba_train_data = torch.utils.data.Subset(celeba_all_aug, train_indices)
        celeba_val_data = torch.utils.data.Subset(celeba_all_aug, val_indices)
        celeba_test_data = torch.utils.data.Subset(celeba_all_eval, test_indices)
    else:
        concept_selection = list(range(0, len(CONCEPT_SEMANTICS)))
        if config.get('selected_concepts', False):
            concept_selection = SELECTED_CONCEPTS
        concept_indices = concept_selection
        concept_names = [CONCEPT_SEMANTICS[i] for i in concept_indices]
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['identity'],
        )
        num_concepts = config.get(
            'num_concepts',
            celeba_train_data.attr.shape[-1],
        )
        vals, counts = np.unique(
            celeba_train_data.identity,
            return_counts=True,
        )
        sorted_labels = list(map(
            lambda x: x[0],
            sorted(zip(vals, counts), key=lambda x: -x[1])
        ))
        # logging.debug(f"Selecting {config['num_classes']} out of {len(vals)} classes")
        result_dir = config.get('result_dir', None)
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                os.path.join(
                    result_dir,
                    f"selected_top_{config['num_classes']}_labels.npy",
                ),
                sorted_labels[:config['num_classes']],
            )
        label_remap = {}
        for i, label in enumerate(sorted_labels[:config['num_classes']]):
            label_remap[label] = i

        # Now reload by transform the labels accordingly
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=False,
            transform=transforms.Compose([
                transforms.Resize(resol),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    # If it is not in our map, then we make it be the token label
                    # config['num_classes'] which will be removed afterwards
                    label_remap.get(
                        x[0].cpu().detach().item() - 1,
                        config['num_classes']
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_selection].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = config['num_classes']

        train_idxs = np.where(
            list(map(
                lambda x: x.cpu().detach().item() - 1 in label_remap,
                celeba_train_data.identity
            ))
        )[0]
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    if not use_binary_vector_class:
        total_samples = len(celeba_train_data)
        train_samples = int(0.7 * total_samples)
        test_samples = int(0.2 * total_samples)
        val_samples = total_samples - test_samples - train_samples
        celeba_train_data, celeba_test_data, celeba_val_data = \
            torch.utils.data.random_split(
                celeba_train_data,
                [train_samples, test_samples, val_samples],
            )
    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    seed_everything(seed) # reset the seed

    # Finally, determine whether or not we will need to compute the imbalance factors
    if config.get('weight_loss', False):
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for i, (_, (y, c)) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    
    return train_dl, val_dl, test_dl, imbalance, (num_concepts, len(label_remap), concept_group_map, concept_names, concept_indices)
