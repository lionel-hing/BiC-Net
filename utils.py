import csv
import ctypes
import datetime
import logging
import multiprocessing as mp
import random
import sys
import argparse
from pathlib import Path
from typing import Union, Tuple, Dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict
from torch import cuda

EVALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
EVALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"


def get_csv_header_keys(compute_clip_retrieval):
    metric_keys = ["ep", "time"]
    prefixes = ["v", "p"]
    if compute_clip_retrieval:
        prefixes += ["c", "s"]
    for prefix in prefixes:
        for key in EVALKEYS:
            metric_keys.append(f"{prefix}-{key}")
    return metric_keys


def print_csv_results(csv_file: str, print_fn=print):
    metric_keys = get_csv_header_keys(False)
    with Path(csv_file).open("rt", encoding="utf8") as fh:
        reader = csv.DictReader(fh, metric_keys)
        line_data = [line for line in reader][1:]
        for line in line_data:
            for key, val in line.items():
                line[key] = float(val)
    relevant_field = [line["v-r10"] + line["p-r10"] for line in line_data]
    best_epoch = np.argmax(relevant_field)

    def get_res(search_key):
        results = {}
        for key_, val_ in line_data[best_epoch].items():
            if key_[:2] == f"{search_key}-":
                results[key_[2:]] = float(val_)
        return results

    print_fn(f"Total epochs {len(line_data)}. "
             f"Results from best epoch {best_epoch}:")
    print_fn(EVALHEADER)
    print_fn(retrieval_results_to_str(get_res("p"), "Par2Vid"))
    print_fn(retrieval_results_to_str(get_res("v"), "Vid2Par"))


def expand_segment(num_frames, num_target_frames, start_frame, stop_frame):
    num_frames_seg = stop_frame - start_frame + 1
    changes = False
    if num_target_frames > num_frames:
        num_target_frames = num_frames
    if num_frames_seg < num_target_frames:
        while True:
            if start_frame > 0:
                start_frame -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
            if stop_frame < num_frames - 1:
                stop_frame += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
    return start_frame, stop_frame, changes


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_config(file: Union[str, Path]) -> EasyDict:
    with Path(file).open("rt", encoding="utf8") as fh:
        config = yaml.load(fh, Loader=yaml.Loader)
    cfg = EasyDict(config)
    # model symmetry
    for check_network in ["net_text_pooler", "net_text_sequencer"]:
        if getattr(cfg, check_network).name == "same":
            setattr(cfg, check_network, getattr(
                cfg, getattr(cfg, check_network).same_as))
    return cfg


def dump_config(cfg: EasyDict, file: Union[str, Path]) -> None:
    with Path(file).open("wt", encoding="utf8") as fh:
        yaml.dump(cfg, fh, Dumper=yaml.Dumper)


def print_config(cfg: EasyDict, level=0) -> None:
    for key, val in cfg.items():
        if isinstance(val, EasyDict):
            print("     " * level, str(key), sep="")
            print_config(val, level=level + 1)
        else:
            print("    " * level, f"{key} - f{val} ({type(val)})", sep="")


def make_shared_array(np_array: np.ndarray) -> mp.Array:
    flat_shape = int(np.prod(np_array.shape))
    shared_array_base = mp.Array(ctypes.c_float, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(np_array.shape)
    shared_array[:] = np_array[:]
    return shared_array


def compute_indices(
        num_frames_orig: int, num_frames_target: int, is_train: bool):
    def round_half_down(array: np.ndarray) -> np.ndarray:
        return np.ceil(array - 0.5)

    if is_train:
        # random sampling during training
        start_points = np.linspace(
            0, num_frames_orig, num_frames_target, endpoint=False)
        start_points = round_half_down(start_points).astype(int)
        offsets = start_points[1:] - start_points[:-1]
        np.random.shuffle(offsets)
        last_offset = num_frames_orig - np.sum(offsets)
        offsets = np.concatenate([offsets, np.array([last_offset])])
        new_start_points = np.cumsum(offsets) - offsets[0]
        offsets = np.roll(offsets, -1)
        random_offsets = offsets * np.random.rand(num_frames_target)
        indices = new_start_points + random_offsets
        indices = np.floor(indices).astype(int)
        return indices
    # center sampling during validation
    start_points = np.linspace(
        0, num_frames_orig, num_frames_target, endpoint=False)
    offset = num_frames_orig / num_frames_target / 2
    indices = start_points + offset
    indices = np.floor(indices).astype(int)
    return indices


def truncated_normal_fill(
        shape: Tuple[int], mean: float = 0, std: float = 1,
        limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def retrieval_results_to_str(results: Dict[str, float], name: str):
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(
        name, *[results[a] for a in EVALKEYS])


def compute_retr_vid_to_par(region_feat, global_feat, cap_feat):
    num_points = global_feat.shape[0]
    d1 = np.dot(global_feat, cap_feat.T)
    d2 = np.dot(region_feat, cap_feat.T)
    fused_d = (0.5 * d1) + (0.5 * d2)
    return compute_retrieval_cosine(fused_d, num_points)


def compute_retr_par_to_vid(region_feat, global_feat, cap_feat):
    num_points = global_feat.shape[0]
    d1 = np.dot(cap_feat, global_feat.T)
    d2 = np.dot(cap_feat, region_feat.T)
    fused_d = (0.5 * d1) + (0.5 * d2)
    return compute_retrieval_cosine(fused_d, num_points)


def compute_retr_vid_to_par_a(global_feat, cap_feat):
    num_points = global_feat.shape[0]
    d = np.dot(global_feat, cap_feat.T)
    return compute_retrieval_cosine(d, num_points)


def compute_retr_par_to_vid_a(global_feat, cap_feat):
    num_points = global_feat.shape[0]
    d = np.dot(cap_feat, global_feat.T)
    return compute_retrieval_cosine(d, num_points)


def compute_retrieval_cosine(dot_product, len_dot_product):
    ranks = np.zeros(len_dot_product)
    top1 = np.zeros(len_dot_product)
    for index in range(len_dot_product):
        inds = np.argsort(dot_product[index])[::-1]
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        top1[index] = inds[0]
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = dict()
    report_dict['r1'] = r1
    report_dict['r5'] = r5
    report_dict['r10'] = r10
    report_dict['r50'] = r50
    report_dict['medr'] = medr
    report_dict['meanr'] = meanr
    report_dict['sum'] = r1 + r5 + r50
    return report_dict, top1, ranks


def get_logging_formatter():
    return logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%m%d %H%M%S")


def get_timestamp_for_filename():
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


def get_logger_without_file(name, log_level="INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(get_logging_formatter())
    logger.addHandler(strm_hdlr)
    return logger


def get_logger(
        logdir, name, filename="run", log_level="INFO", log_file=True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = get_logging_formatter()
    if log_file:
        file_path = Path(logdir) / "{}_{}.log".format(
            filename, str(datetime.datetime.now()).split(".")[0].replace(
                " ", "_").replace(":", "_").replace("-", "_"))
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(formatter)
        logger.addHandler(file_hdlr)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)
    logger.addHandler(strm_hdlr)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger):
    x = list(logger.handlers)
    for i in x:
        logger.removeHandler(i)
        i.flush()
        i.close()


def get_args(description='VSR-Net for text-video retrieval task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, default="msr-vtt", help="abbreviation of dataset")
    parser.add_argument("--data_split", type=str, default="official", help="")
    parser.add_argument("--workers", type=int, default=None, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--log_dir", type=str, default="/workspace/ninghan/MSR-VTT"
                                                       "/runs", help="")
    parser.add_argument("--dataroot", type=str, default="/workspace/ninghan"
                                                        "/MSR-VTT", help="")
    parser.add_argument("--cuda", action="store_true", help="")
    parser.add_argument("--preload_vid", action="store_true", help="Load video features into RAM")
    parser.add_argument("--no_preload_text", action="store_true", help="Do not load text features into RAM")
    parser.add_argument("--checkpoint", type=str, default="", help="the checkpoint path for eval")
    parser.add_argument("--is_train", action="store_true", help="set true for train")

    # model param
    parser.add_argument("--global_dim", type=int, default=2560, help="the dimension of global video feature")
    parser.add_argument("--region_dim", type=int, default=2048, help="the dimension of region feature")
    parser.add_argument("--text_dim", type=int, default=1536, help="the dimension of text feature")
    parser.add_argument("--num_heads", type=int, default=16, help="the number of heads ")
    parser.add_argument("--region_num", type=int, default=36, help="the number of region ")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="dimension of joint embedding space")
    parser.add_argument("--layer_num", type=int, default=4, help="the number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mlp_ration", type=int, default=2, help="mlp rate")
    parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout rate")

    args = parser.parse_args()

    return args
