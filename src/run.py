# -*- coding: utf-8 -*-
"""
SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts

Copyright (c) 2024 Gengze Zhou, Yicong Hong, Zun Wang, Chongyang Zhao, Mohit Bansal, Qi Wu

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Author: Gengze Zhou
Email: gengze.zhou@adelaide.edu.au
Paper: https://arxiv.org/abs/2412.05552
"""

import os
import yaml
import random
import datetime
import argparse
import numpy as np
import torch

from omegaconf import OmegaConf
from utils.distributed import world_info_from_env, init_distributed_device
from utils.common_utils import setup_logger, log_config_to_file
from tasks.loaders import create_dataloaders, create_environments
from tasks.agents import load_agent
from trainers import load_trainer


def setup_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment runner with OmegaConf")

    parser.add_argument('--config_dir', type=str, default='configs/main_multi_q.yaml', help="Path to the experiment config file")
    # Option to override specific configurations using key-value pairs
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the config. Use the key-value pair in xxx=yyy format. Example: --options training.learning_rate=0.01"
    )

    args = parser.parse_args()

    # Load the default configuration
    default_config_path = 'configs/default.yaml'
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")
    default_config = OmegaConf.load(default_config_path)

    # Load the experiment-specific configuration
    if not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Experiment config file not found: {args.config_dir}")
    experiment_config = OmegaConf.load(args.config_dir)

    # Merge default config with experiment config
    config = OmegaConf.merge(default_config, experiment_config)

    # Apply overrides using the options argument
    if args.options:
        for kv in args.options:
            key, value = kv.split("=", 1)
            # value is a string, try to convert it to the correct type
            if value in ['True', 'False']:
                value = True if value == 'True' else False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            OmegaConf.update(config, key, value)

    return config


def main():
    ############################################################
    # Parse arguments and load configuration
    ############################################################

    config = parse_args()

    # Set up environment variables for distributed training
    config.distributed.local_rank, config.distributed.rank, config.distributed.world_size = world_info_from_env()
    device_id, is_distributed = init_distributed_device(config.distributed)

    # Setup logging
    output_path = os.path.join(config.experiment.output_dir, config.experiment.id)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'results'), exist_ok=True)
    logger = setup_logger(log_file=os.path.join(output_path, f'{config.experiment.id}.log'), rank=config.distributed.rank)

    # Log configuration (from OmegaConf)
    logger.info(f'********************** Start logging **********************')
    log_config_to_file(config, logger=logger)

    # Ramdom seed setting
    setup_seeds(seed=config.experiment.seed)

    ############################################################
    # Set up data loaders and agents
    ############################################################
    logger.info('********************** Setting up dataloaders **********************')

    environments = create_environments(config, logger)

    # Create train and validation data loaders
    if not config.experiment.test:
        train_dataloaders = create_dataloaders(
            config, logger, environments,
            training=True, device=device_id
        )
    else:
        train_dataloaders = None
    val_dataloaders = create_dataloaders(
        config, logger, environments,
        training=False, device=device_id
    )

    # Create agent
    logger.info(f"Building {config.agent.type} agent...")
    agent = load_agent(
        name=config.agent.type,
        args=config,
        envs=environments,
        device=device_id
    )

    ############################################################    
    # Training
    ############################################################

    # Set up tensorboard writer or Naptune.ai logger
    neptune_api_token = os.getenv('NEPTUNE_API_TOKEN') if os.getenv('NEPTUNE_API_TOKEN') else config.experiment.neptune_api_token
    if config.experiment.use_neptune and neptune_api_token:
        import neptune
        writer = neptune.init_run(
            project=config.experiment.neptune_project,
            api_token=neptune_api_token,
            name=config.experiment.id,
            tags=list(config.experiment.tags),
            monitoring_namespace="monitoring"
        )
        writer['config'] = OmegaConf.to_container(config)
    else:
        config.experiment.use_neptune = False
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard'))

    trainer = load_trainer(config.training.trainer, config, agent, train_dataloaders, val_dataloaders, logger, writer)

    if config.experiment.test:
        trainer.test()
    else:
        if config.experiment.eval_first:
            trainer.val_one_epoch(epoch=0)
        trainer.train()



if __name__ == "__main__":
    main()