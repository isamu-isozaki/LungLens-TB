import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from datasets import load_dataset
from diffusers.optimization import get_scheduler

import wandb
import timm
from timm.models import ResNet, ConvNeXt, EfficientNet, DenseNet
from timm.layers import NormMlpClassifierHead, ClassifierHead, create_classifier, LayerNorm2d, LayerNorm
from torchvision import datasets

from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

logger = get_logger(__name__)


def  load_model(model, model_path, accelerator, *args, **kwargs):
    """
    Load a model from a saved state dictionary.

    Args:
        model (class): Model
        model_path (str): Path to the saved model state dictionary.
        accelerator (Accelerator): An accelerator object from the `accelerate` library.

    Returns:
        torch.nn.Module: The loaded and possibly accelerated model.
    """
    logger.info("Loading model")
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=accelerator.device)
    # state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # Use accelerator.prepare_model to wrap the model for distributed/parallel processing
    return model

def log_validation(model, accelerator, val_dataloader):
    logger.info("Logging validation")
    
    v_model = accelerator.unwrap_model(model)
    
    predicted_list, target_list = [],[]

    # Get v_model to evaluation mode
    v_model.eval()
    with torch.no_grad():
        running_loss = 0
        for batch in tqdm(val_dataloader):
            image = batch["pixel_values"].to(device=accelerator.device)
            target = batch["labels"].to(device=accelerator.device).long()
            predicted = v_model(image)
            loss = F.binary_cross_entropy_with_logits(predicted.float(), target.float(), reduction="mean")
            running_loss+=loss
            
            predicted_list.extend(torch.argmax(predicted, dim=1).cpu().detach().numpy().tolist())
            target_list.extend(torch.argmax(target, dim=1).cpu().detach().numpy().tolist())

    # Compute metrics
    avg_loss = running_loss / len(val_dataloader)
    accuracy = accuracy_score(target_list, predicted_list)
    f1 = f1_score(target_list, predicted_list)

    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    return avg_loss, accuracy, f1


def save_progress(model, model_path, accelerator):
    logger.info("Saving model")
    torch.save(accelerator.unwrap_model(model).state_dict(), model_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script.")
    parser.add_argument(
        "--freeze_features",
        action="store_true",
        help="Freeze everything except base in timm model.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--timm_model_name",
        type=str,
        default="resnet50",
        help="Name of timm model.",
    )
    parser.add_argument(
        "--train_dataset", type=str, default="alkzar90/NIH-Chest-X-ray-dataset", help="Path to hf dataset."
    )
    parser.add_argument(
        "--train_configuration", type=str, default="image-classification", help="Either image-classification or object-detection."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lunglens-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


class TBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert label to a list to match your specified format
        return {'image': image, 'labels': [label]}


class NIHDataset(Dataset):
    def __init__(
        self,
        dataset,
        transform,
    ):
        self.dataset = dataset
        self.num_images = len(self.dataset)
        self._length = self.num_images

        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        element = self.dataset[i % self.num_images]
        image = element["image"]
        if element["labels"] == [0]:
            label = [1, 0]
        else:
            label = [0, 1]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["pixel_values"] = self.transform(image)
        example["labels"] = label
        return example
    
def collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    labels = [torch.tensor(example["labels"]).long() for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    labels = torch.stack(labels)

    batch = {
        "pixel_values": pixel_values,
        "labels": labels,
    }

    return batch

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    model = timm.create_model(args.timm_model_name, pretrained=True)

    # for all global pool below I assumed it was average
    if isinstance(model, ConvNeXt):
        if isinstance(model.head, ClassifierHead):
            model.head = ClassifierHead(
                model.num_features,
                2,
                pool_type="avg",
                drop_rate=model.drop_rate
            )
        else:
            head_hidden_size = None
            if "convnext_large_mlp" in args.timm_model_name:
                head_hidden_size = 1536
            norm_layer = LayerNorm2d
            model.head = NormMlpClassifierHead(
                model.num_features,
                2,
                hidden_size=head_hidden_size,
                pool_type="avg",
                drop_rate=model.drop_rate,
                norm_layer=norm_layer,
                act_layer='gelu',
            )
        head = model.head
    elif isinstance(model, ResNet):
        model.global_pool, model.fc = create_classifier(model.num_features, 2, pool_type="avg")
        head = model.fc
    elif isinstance(model, EfficientNet) or isinstance(model, DenseNet):
        model.global_pool, model.classifier = create_classifier(model.num_features, 2, pool_type="avg")
        head = model.classifier
    else:
        raise NotImplementedError(f"{args.timm_model_name} logic is not implemented")
    
    # Load the best model for finetuning
    model = load_model(model, 'lunglens-model/best_model_val.bin', accelerator)

    if args.freeze_features:
        model.requires_grad_(False)
        head.requires_grad_(True)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    # warning: below does centercropping
    transform = timm.data.create_transform(**data_cfg)

    

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        head.parameters() if args.freeze_features else model.parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    class TBDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            return {'image': image, 'labels': [label]}

    # Load the dataset
    dataset_path = '../Data/TB_Chest_Radiography_Database/'
    original_dataset = datasets.ImageFolder(root=dataset_path)
    custom_dataset = TBDataset(original_dataset)

    # Get targets for stratification
    targets = [label for _, label in original_dataset.imgs]

    # Split into train and temp (temporary will be split further into validation and test)
    train_indices, temp_indices = train_test_split(
        range(len(custom_dataset)),
        test_size=0.4,  # Suppose we reserve 40% of data initially to split further into validation and test
        stratify=targets,
        random_state=42
    )

    # Now split the temp dataset into validation and test datasets
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # Split the temporary set evenly into validation and test sets
        stratify=[targets[i] for i in temp_indices],  # Stratify according to the subset of targets
        random_state=42
    )

    # Create train, validation, and test subsets
    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)
    test_dataset = Subset(custom_dataset, test_indices)

    # Dataset and DataLoaders creation:
    train_dataset = NIHDataset(
        dataset=train_dataset,
        transform=transform
    )

    val_dataset = NIHDataset(
        dataset=val_dataset,
        transform=transform
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, collate_fn=collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, collate_fn=collate_fn
    )

    # if args.validation_epochs is not None:
    #     warnings.warn(
    #         f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
    #         " Deprecated validation_epochs in favor of `validation_steps`"
    #         f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
    #         FutureWarning,
    #         stacklevel=2,
    #     )
    #     args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    model.train()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    model.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("lunglens_tb", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    best_val_loss = 1e100
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                image = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                target = batch["labels"].to(device=accelerator.device).long()
                predicted = model(image)
                loss = F.binary_cross_entropy_with_logits(predicted.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"FineTuned-{args.timm_model_name}-{global_step}.bin"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        model, save_path, accelerator
                    )

                logs = {"training loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
                
                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        val_loss, val_accuracy, val_f1 = log_validation(
                            model, accelerator, val_dataloader
                        )

                        if val_loss<best_val_loss:
                            best_model_name = (
                                f"ft_best_model_val.bin"
                            )
                            best_save_path = os.path.join(args.output_dir, best_model_name)

                            save_progress(
                                model, best_save_path, accelerator
                            )
                            best_val_loss = val_loss
                        
                        model.train()

                        logs = {"training loss": loss.detach().item(), 
                                "validation loss": val_loss,
                                "validation accuracy": val_accuracy,
                                "validation f1": val_f1,
                                "lr": lr_scheduler.get_last_lr()[0]}
            
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()