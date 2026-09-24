# MaskLAT

MaskLAT is a vision-language segmentation architecture that maintains a compact latent table alongside decoder queries. The repository provides model definitions and configurations for Phi-3 and Qwen3 language backbones.

## Architecture

The segmentation decoder produces query states and masks at every stage. The first three stages use independent grouped proposal builders:

1. ST1 groups mask proposals and initializes the latent table.
2. ST2 refines the ST1 latent table with its own query and mask predictions.
3. ST3 refines the ST2 output and projects the final prefix latents into the language model.

Only the ST3 latent output is inserted into the language sequence. Decoder queries are unchanged by the three prefix builders.

After the language model, stages ST4–ST9 exchange information between segmentation queries and persistent latents. Each stage forms supervision groups from its own mask predictions. A final condition-to-latent attention block updates the condition representation used by the classification head.

The grouped objective matches latent attention distributions to query groups derived from mask topology. It is added to the standard segmentation and classification objectives without replacing them.

## Backbones

- Phi-3: `masklat/configs/models/phi3_grouped_st123.py`
- Qwen3: `masklat/configs/models/qwen3_grouped_st123.py`

Both configurations implement the same segmentation and latent pathway. They differ only in the language backbone and its text-context configuration.

## Repository layout

```text
masklat/
  dataset/       datasets, processors, samplers, and collation
  engine/        training hooks and runner utilities
  evaluation/    segmentation evaluators and metrics
  model/         vision-language model and grouped latent modules
  configs/       Phi-3 and Qwen3 model configurations
  tools/         training and evaluation entry points
tests/           unit and integration contracts
configs/         distributed-training runtime configurations
```

This repository contains source code and configuration files only. Model checkpoints and datasets are not included.

