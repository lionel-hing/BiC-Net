
from pathlib import Path
import os
import utils
from dataset import create_datasets, create_loaders
from trainer import TrainerVideoText

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"


def main():
    args = utils.get_args()
    utils.set_seed(args.seed)
    num_workers = 4 if args.workers is None else args.workers
    print(f"{num_workers} parallel dataloader workers")
    dataset_path = Path(args.dataroot)
    train_set, val_set, test_set = create_datasets(dataset_path, args, False, False)
    train_loader, val_loader, test_loader = create_loaders(train_set, val_set, test_set, args.batch_size, num_workers)

    trainer = TrainerVideoText(args)
    trainer.validate(test_loader)
    trainer.close()


if __name__ == "__main__":
    main()
