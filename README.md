# NF-Classifier
Normalizing flow based classifiers
## Description ##
Playground for normalizing flow based classifiers. Given a labeled dataset, features are treated as conditional inputs to NF and the intention is to learn distribution for labels.

The backbone of the NF is implemented using MAF and the head is implemented using argmax flow. 

## Usage ##
```python
usage: main.py [-h] [--task TASK] [--num_blocks NUM_BLOCKS]
               [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE]
               [--weight_decay WEIGHT_DECAY] [--augment_noise]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Task name.
  --num_blocks NUM_BLOCKS
                        Number of epochs to train.
  --num_epochs NUM_EPOCHS
                        Number of epochs to train.
  --learning_rate LEARNING_RATE
                        Initial learning rate for the trainer.
  --weight_decay WEIGHT_DECAY
                        Weight decay for the trainer.
  --augment_noise       Whether to enable inverse multi-scale.

```

## Refererences ##
MAF: https://arxiv.org/abs/1705.07057 \
SurVAE Flows: https://arxiv.org/abs/2007.02731 \
Argmax Flows: https://arxiv.org/abs/2102.05379 \
Densely Connected NF: https://arxiv.org/abs/2106.04627
