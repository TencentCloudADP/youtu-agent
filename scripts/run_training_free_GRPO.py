#!/usr/bin/env python3
"""Command line interface for Training-free GRPO."""

import asyncio

from utu.practice import TrainingFreeGRPO, parse_training_free_grpo_config


async def main():
    """Run TrainingFreeGRPO from command line."""
    config = parse_training_free_grpo_config()
    training_free_grpo = TrainingFreeGRPO(config)
    result = await training_free_grpo.run()
    print(f"Training-free GRPO completed. New agent config file with experiences saved at: {result}")


if __name__ == "__main__":
    asyncio.run(main())
