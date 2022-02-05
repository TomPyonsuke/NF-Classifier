import argparse
from cifar10 import train_cifar10

def main(args):
    if args.task == 'cifar10':
        train_cifar10(
            n_epochs=args.num_epochs,
            n_blocks=args.num_blocks,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            augment_noise=args.augment_noise
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        help='Task name.',
                        default='cifar10')
    parser.add_argument('--num_blocks',
                        type=int,
                        default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Initial learning rate for the trainer.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay for the trainer.')
    parser.add_argument('--augment_noise',
                        action='store_true',
                        help='Whether to enable inverse multi-scale.')


    args = parser.parse_args()
    main(args)