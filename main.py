import arguments

from src.trainer import Trainer


if __name__ == '__main__':
    parser = arguments.parse_args()
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
