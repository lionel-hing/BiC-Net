import os
from pathlib import Path
from multiprocessing import cpu_count
import transformers
import utils
from dataset import create_datasets, create_loaders
from trainer import TrainerVideoText

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def main():
    args = utils.get_args()
    data_file = Path(args.dataroot)
    num_workers = min(1, cpu_count() - 1)
    print(f"{num_workers} parallel Dataloader workers")
    if args.seed != -1:
        utils.set_seed(args.seed)
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                           TOKENIZERS_PARALLELISM=False)
    train_set, val_set, test_set = create_datasets(data_file, args)
    train_loader, val_loader, _ = create_loaders(train_set, val_set, test_set, args.batch_size, num_workers)

    trainer = TrainerVideoText(args, tokenizer)
    trainer.train_loop(train_loader, val_loader)
    trainer.logger.info("---------- Results ----------")
    utils.print_csv_results(trainer.log_dir / "train_metrics.csv", print_fn=trainer.logger.info)
    trainer.close()


if __name__ == '__main__':
    main()
