import json
from pathlib import Path
import pandas as pd
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import utils


class BertTextFeatureLoader:
    """collect text data processed from pre-trained bert"""
    def __init__(self, dataset_path, ids, data_name: str = "default", preload=True):
        self.h5_path = (dataset_path / f"text_{data_name}.h5")
        lens_file = (dataset_path / f"text_lens_{data_name}.json")
        self.lens = json.load(lens_file.open("rt", encoding="utf8"))
        self.cached_data = None
        if preload:
            h5file = h5py.File(self.h5_path, "r")
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload text..."):
                np_array = h5file[id_]
                shared_array = utils.make_shared_array(np_array)
                self.cached_data[id_] = shared_array
            h5file.close()

    def __getitem__(self, id_):
        lens = self.lens[id_]
        if self.cached_data is None:
            h5file = h5py.File(self.h5_path, "r")
            features = np.array(h5file[id_])
            h5file.close()
            return features, lens
        return self.cached_data[id_], lens


class TextLoader:
    """collect text data"""
    def __init__(self, dataset_path, ids, data_name: str = "default", preload=True):
        self.text_path = (dataset_path / f"datainfo.json")
        lens_file = (dataset_path / f"text_lens_{data_name}.json")
        self.lens = json.load(lens_file.open("rt", encoding="utf8"))
        self.cached_data = None
        if preload:
            with open(self.text_path, 'r') as fid:
                json_data = json.load(fid)
            text_data_from_json = json_data["sentences"]
            text_data = {}
            for text in text_data_from_json:
                id_ = text["video_id"]
                if id_ not in text_data.keys():
                    text_data[id_] = [text["caption"]]
                else:
                    text_data[id_].append(text["caption"])
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload text..."):
                self.cached_data[id_] = text_data[id_]

    def __getitem__(self, id_):
        lens = self.lens[id_]
        if self.cached_data is None:
            h5file = h5py.File(self.text_path, "r")
            features = np.array(h5file[id_])
            h5file.close()
            return features, lens
        return self.cached_data[id_], lens


class MsrvttVideoFeatureLoader:
    """Featureloader for MSR-VTT dataset"""
    def __init__(self, dataset_path, ids, preload=False):
        dataset_path = Path(dataset_path)
        self.h5_path_region = dataset_path / f"msrvtt_region_feature.h5"
        self.h5_path_global = dataset_path / f"msr-vtt_features.h5"
        self.cached_region_data = None
        self.cached_global_data = None

        h5_region = h5py.File(self.h5_path_region, 'r')
        self.region_feats = h5_region['vfeats']
        h5_global = h5py.File(self.h5_path_global, 'r')
        self.global_feats = h5_global['feats']

        if preload:
            self.cached_region_data = {}
            self.cached_global_data = {}
            for id_ in tqdm(ids, desc="preload videos..."):
                video_num = int(id_[5:])
                np_array_region = self.region_feats[video_num]
                np_array_global = self.global_feats[video_num]
                shared_array_region = utils.make_shared_array(np_array_region)
                shared_array_global = utils.make_shared_array(np_array_global)
                self.cached_region_data[id_] = shared_array_region
                self.cached_global_data[id_] = shared_array_global
            h5_region.close()
            h5_global.close()

    def __getitem__(self, id_):
        if self.cached_global_data is None and self.cached_region_data is None:
            region_feat = np.array(self.region_feats[id_])
            global_feat = np.array(self.global_feats[id_])
            return region_feat, global_feat
        else:
            return self.cached_region_data[id_], self.cached_global_data[id_]


class MSVDVideoFeatureLoader:
    """Featureloader for MSVD dataset"""
    def __init__(self, dataset_path, ids, preload=False):
        dataset_path = Path(dataset_path)
        self.h5_path_region = dataset_path / f"msvd_region_feature.h5"
        self.h5_path_global = dataset_path / f"msvd_features.h5"
        self.cached_region_data = None
        self.cached_global_data = None

        # test
        h5_region = h5py.File(self.h5_path_region, 'r')
        self.region_feats = h5_region['vfeats']
        h5_global = h5py.File(self.h5_path_global, 'r')
        self.global_feats = h5_global['feats']

        if preload:
            self.cached_region_data = {}
            self.cached_global_data = {}
            for id_ in tqdm(ids, desc="preload videos..."):
                video_num = int(id_[5:])
                np_array_region = self.region_feats[video_num]
                np_array_global = self.global_feats[video_num]
                shared_array_region = utils.make_shared_array(np_array_region)
                shared_array_global = utils.make_shared_array(np_array_global)
                self.cached_region_data[id_] = shared_array_region
                self.cached_global_data[id_] = shared_array_global
            h5_region.close()
            h5_global.close()

    def __getitem__(self, id_):
        if self.cached_global_data is None and self.cached_region_data is None:
            region_feat = np.array(self.region_feats[id_])
            global_feat = np.array(self.global_feats[id_])
            return region_feat, global_feat
        else:
            return self.cached_region_data[id_], self.cached_global_data[id_]


class VideoDatasetFeatures(data.Dataset):
    """Dataloader for extracted video feature"""
    def __init__(self, dataset_name, dataset_path, split, data_split, is_train, preload_vid_feat, preload_text_feat):
        self.split = split
        self.is_train = is_train
        if dataset_name == "msr-vtt":
            if split == "train":
                vids_id_file = Path(dataset_path) / f"MSRVTT_{data_split}_train.csv"
                csv_file = pd.read_csv(vids_id_file)
                self.ids = list(csv_file["video_id"])
            elif split == "validate":
                if data_split == "official":
                    vids_id_file = Path(dataset_path) / f"MSRVTT_{data_split}_validate.csv"
                    csv_file = pd.read_csv(vids_id_file)
                    self.ids = list(csv_file["video_id"])
                else:
                    vids_id_file = Path(dataset_path) / f"MSRVTT_{data_split}_test.csv"
                    csv_file = pd.read_csv(vids_id_file)
                    self.ids = list(csv_file["video_id"])
            else:
                vids_id_file = Path(dataset_path) / f"MSRVTT_{data_split}_test.csv"
                csv_file = pd.read_csv(vids_id_file)
                self.ids = list(csv_file["video_id"])
        elif dataset_name == "msvd":
            train_list = list((range(0, 1200)))
            train_list = ["video" + str(i) for i in train_list]
            val_list = list((range(1200, 1300)))
            val_list = ["video" + str(i) for i in val_list]
            test_list = list((range(1300, 1970)))
            test_list = ["video" + str(i) for i in test_list]
            if split == "train":
                self.ids = train_list
            elif split == "validate":
                self.ids = val_list
            else:
                self.ids = test_list
        else:
            raise NotImplementedError
        print(f"init dataset {dataset_name} split {split} length {len(self.ids)} ")
        if dataset_name == "msvd":
            self.raw_text_data = TextLoader(
                dataset_path, self.ids, "default", preload_text_feat)
            self.text_data = BertTextFeatureLoader(
                dataset_path, self.ids, "default", preload_text_feat)
            self.vid_data = MSVDVideoFeatureLoader(
                dataset_path, self.ids, preload_vid_feat)
        elif dataset_name == "msr-vtt":
            self.raw_text_data = TextLoader(
                dataset_path, self.ids, "default", preload_text_feat)
            self.text_data = BertTextFeatureLoader(
                dataset_path, self.ids, "default", preload_text_feat)
            self.vid_data = MsrvttVideoFeatureLoader(
                dataset_path, self.ids, preload_vid_feat)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # need to return text length for making mask
        # text-part
        vid_id = self.ids[index]
        text_lens = self.text_data.lens[vid_id]
        para_len = sum(text_lens)
        if self.text_data.cached_data is not None:
            para_feats = self.text_data.cached_data[vid_id]
        else:
            para_feats, _ = self.text_data[vid_id]
        para_feats = torch.tensor(para_feats).float()
        # take multi sentence feats
        text_feats_list = []
        sent_lens = 0
        sent_num_list = random.sample(range(0, len(text_lens) - 1), 1)
        for sent_num in sent_num_list:
            sent_len = text_lens[sent_num]
            sent_start = sum(text_lens[:sent_num])
            sent_end = sent_start + sent_len
            sent_feats = para_feats[sent_start:sent_end]
            text_feats_list.append(sent_feats)
            sent_lens += sent_len
        multi_text_feats = torch.cat(text_feats_list, dim=0)

        # take single sentence feats
        single_sent_num = random.randint(0, len(text_lens) - 1)
        single_sent_lens = text_lens[single_sent_num]
        single_sent_start = sum(text_lens[:single_sent_num])
        single_sent_end = single_sent_start + single_sent_lens
        single_sent_feats = para_feats[single_sent_start:single_sent_end]

        # video-part
        vid_num = vid_id[5:]
        vid_num = int(vid_num)
        region_feats = torch.tensor(())
        region_feats, global_feats = self.vid_data[vid_num]
        region_feats = torch.tensor(region_feats)
        global_feats = torch.tensor(global_feats)
        frame_lens = region_feats.shape[0]
        region_lens = region_feats.shape[1]

        # raw_text-part
        raw_texts = None
        if self.raw_text_data.cached_data is not None:
            raw_texts = self.raw_text_data.cached_data[vid_id]
            raw_texts = random.choice(raw_texts)

        return {
            "text_feats": multi_text_feats,
            "text_lens": sent_lens,
            "sent_feats": single_sent_feats,
            "sent_lens": single_sent_lens,
            "region_feats": region_feats,
            "global_feats": global_feats,
            "frame_lens": frame_lens,
            "region_lens": region_lens,
            "video_id": vid_id,
            "raw_texts": raw_texts
        }

    def collate_fn(self, data_batch):
        def get_data(key):
            return [d[key] for d in data_batch]

        batch_size = len(data_batch)

        # collate video global data
        list_global_feats = get_data("global_feats")
        list_frames_lens = get_data("frame_lens")
        global_feats_dim = list_global_feats[0].shape[-1]
        frames_lens = torch.tensor(list_frames_lens).long()
        frames_max_len = int(frames_lens.max().numpy())
        global_feats = torch.zeros(
            batch_size, frames_max_len, global_feats_dim).float()
        global_mask = torch.ones(batch_size, frames_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_frames_lens, list_global_feats)):
            global_feats[batch, :seq_len, :] = item

        # collate video region data
        list_region_feats = get_data("region_feats")
        list_region_lens = get_data("region_lens")
        region_feats_dim = list_region_feats[0].shape[-1]
        region_lens = torch.tensor(list_region_lens).long()
        region_max_len = int(region_lens.max().numpy())
        region_feats = torch.zeros(
            batch_size, frames_max_len, region_max_len, region_feats_dim).float()
        region_mask = torch.ones(batch_size, frames_max_len, region_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_region_lens, list_region_feats)):
            region_feats[batch, :, :, :] = item

        # collate text region data
        list_text_len = get_data("text_lens")
        list_text_feats = get_data("text_feats")
        text_feats_dim = list_text_feats[0].shape[-1]
        text_len = torch.tensor(list_text_len).long()
        text_max_len = int(text_len.max().numpy())
        text_feats = torch.zeros(
            batch_size, text_max_len, text_feats_dim).float()
        text_mask = torch.zeros(
            batch_size, text_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_text_len, list_text_feats)):
            text_feats[batch, :seq_len, :] = item
            text_mask[batch, :seq_len] = 1

        video_id = get_data("video_id")

        # collate raw text data
        raw_text_data = get_data("raw_texts")

        return {
            "text_feats": text_feats,
            "text_mask": text_mask,
            "region_feats": region_feats,
            "region_mask": region_mask,
            "global_feats": global_feats,
            "global_mask": global_mask,
            "video_id": video_id,
            "raw_texts": raw_text_data
        }

    def collate_fn_test(self, data_batch):
        def get_data(key):
            return [d[key] for d in data_batch]

        batch_size = len(data_batch)

        # collate video global data
        list_global_feats = get_data("global_feats")
        list_frames_lens = get_data("frame_lens")
        global_feats_dim = list_global_feats[0].shape[-1]
        frames_lens = torch.tensor(list_frames_lens).long()
        frames_max_len = int(frames_lens.max().numpy())
        global_feats = torch.zeros(
            batch_size, frames_max_len, global_feats_dim).float()
        global_mask = torch.ones(batch_size, frames_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_frames_lens, list_global_feats)):
            global_feats[batch, :seq_len, :] = item

        # collate video region data
        list_region_feats = get_data("region_feats")
        list_region_lens = get_data("region_lens")
        region_feats_dim = list_region_feats[0].shape[-1]
        region_lens = torch.tensor(list_region_lens).long()
        region_max_len = int(region_lens.max().numpy())
        region_feats = torch.zeros(
            batch_size, frames_max_len, region_max_len, region_feats_dim).float()
        region_mask = torch.ones(batch_size, frames_max_len, region_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_region_lens, list_region_feats)):
            region_feats[batch, :, :, :] = item

        # collate text region data
        list_text_len = get_data("sent_lens")
        list_text_feats = get_data("sent_feats")
        text_feats_dim = list_text_feats[0].shape[-1]
        text_len = torch.tensor(list_text_len).long()
        text_max_len = int(text_len.max().numpy())
        text_feats = torch.zeros(
            batch_size, text_max_len, text_feats_dim).float()
        text_mask = torch.zeros(
            batch_size, text_max_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_text_len, list_text_feats)):
            text_feats[batch, :seq_len, :] = item
            text_mask[batch, :seq_len] = 1

        video_id = get_data("video_id")

        # collate raw text data
        raw_text_data = get_data("raw_texts")

        return {
            "text_feats": text_feats,
            "text_mask": text_mask,
            "region_feats": region_feats,
            "region_mask": region_mask,
            "global_feats": global_feats,
            "global_mask": global_mask,
            "video_id": video_id,
            "raw_texts": raw_text_data
        }


def create_datasets(dataset_path, args, preload_vid_feat=False, preload_text_feat=True):
    """create dataset for training and validation"""
    train_set = VideoDatasetFeatures(args.dataset, dataset_path, "train", args.data_split, True, preload_vid_feat, preload_text_feat)
    val_set = VideoDatasetFeatures(args.dataset, dataset_path, "test", args.data_split, False, preload_vid_feat, preload_text_feat)
    test_set = VideoDatasetFeatures(args.dataset, dataset_path, "test", args.data_split, False, False, False)
    return train_set, val_set, test_set


def create_loaders(train_set, val_set, test_set, batch_size, num_workers):
    """create dataloader for dataset"""

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=train_set.collate_fn, pin_memory=True)
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=val_set.collate_fn_test, pin_memory=True)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=test_set.collate_fn_test, pin_memory=True)
    return train_loader, val_loader, test_loader
