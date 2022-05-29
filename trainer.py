import csv
import os
from pathlib import Path
from timeit import default_timer as timer
import torch
import copy
import torch.nn.parallel
from torch.nn import functional as F
import utils
from loss import ContrastiveLoss
from model import VSR


def unpack_data(data_dict, use_cuda, device):
    def to_device(x):
        if use_cuda and isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    return [
        to_device(data_dict[a]) for a in
        ("text_feats", "text_mask", "region_feats", "region_mask", "global_feats", "global_mask", "video_id")]


class TrainerVideoText:
    def __init__(self, args, tokenizer):
        self.use_cuda = args.cuda
        self.log_dir = Path(args.log_dir)

        self.timer_start_train = 0
        self.det_best_field_current = 0
        self.det_best_field_best = 0
        self.best_epoch = 0
        self.epochs = args.epochs
        self.layer_num = args.layer_num
        self.tokenizer = tokenizer

        # logger / metrics
        self.metrics_fh = None
        if args.is_train:
            os.makedirs(self.log_dir, exist_ok=True)
            metrics_file = self.log_dir / f"train_metrics.csv"
            metric_keys = utils.get_csv_header_keys(False)
            self.metrics_fh = metrics_file.open("wt", encoding="utf8")
            self.metrics_writer = csv.DictWriter(self.metrics_fh, metric_keys)
            self.metrics_writer.writeheader()
            self.metrics_fh.flush()
        self.logger = utils.get_logger(self.log_dir, "trainer", log_file=args.is_train)

        # build model
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        device_ids = [0, 1, 2, 3]
        self.model = torch.nn.DataParallel(VSR(args), device_ids=device_ids)
        # self.model = VSR(args)
        self.model.to(self.device)
        self.best_model_ckpt = copy.deepcopy(self.model.state_dict())

        # initialize loss function and optimizer
        self.criterion = ContrastiveLoss(args.cuda, max_violation=False)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        # scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5,
                                                                       patience=5, cooldown=5)

        if args.checkpoint != "":
            self.logger.info(f"Load checkpoint {args.checkpoint}")
            self.model = torch.nn.DataParallel(VSR(args))
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(args.checkpoint), False)

    def compare_metrics(self, comparison, best):
        if best is None:
            return True
        threshold = 1e-4
        rel_epsilon = threshold + 1
        return comparison > best * rel_epsilon

    def close(self):
        if self.metrics_fh is not None:
            self.metrics_fh.close()
            utils.close_logger(self.logger)

    def train_loop(self, train_loader, val_loader):
        max_step = len(train_loader)
        self.timer_start_train = timer()
        epoch = 0

        # run epochs
        for epoch in range(0, self.epochs):
            self.model.train()

            # train one epoch
            self.logger.info(
                "---------- Training epoch {} ----------".format(
                    epoch))
            for step, data_dict in enumerate(train_loader):
                (text_feats, text_mask, region_feats, region_mask,
                 global_feats, global_mask, video_id) = unpack_data(data_dict, self.use_cuda, device=self.device)

                # forward pass
                (text_emb, region_emb, global_emb) = self.model.forward(text_feats, region_feats,
                                                                        global_feats, text_mask,
                                                                        region_mask, global_mask)

                loss1 = self.criterion(global_emb, text_emb, self.device)
                loss2 = self.criterion(region_emb, text_emb, self.device)
                loss = (0.5 * loss1) + (0.5 * loss2)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logging
                if step % 10 == 0:
                    el_time = (timer() - self.timer_start_train) / 60
                    l_ms = len(str(max_step))
                    str_step = ("{:" + str(l_ms) + "d}").format(step)
                    print_string = (
                        f"E{epoch}[{str_step}/{max_step}] T {el_time:.3f}m "
                        f"LR {self.optimizer.param_groups[0]['lr']:5.3e} "
                        f"L1 {loss1:.5f} "
                        f"L2 {loss2:.5f} "
                        f"L {loss:.5f}")
                    self.logger.info(print_string)

            # validate one epoch
            self.logger.info(
                "---------- Validating epoch {} ----------".format(epoch))
            vid_metrics, clip_metrics = self.validate(val_loader)
            v2p_res, p2v_res, vid_best_at_1 = vid_metrics
            c2s_res, s2c_res, clip_best_at_1 = None, None, None

            # find field which determines is_best
            self.det_best_field_current = vid_best_at_1

            # check if best
            is_best = self.compare_metrics(self.det_best_field_current, self.det_best_field_best)

            if is_best:
                self.det_best_field_best = self.det_best_field_current
                self.best_epoch = epoch

            # write validation results to csv
            csv_input = {
                "ep": epoch,
                "time": timer() - self.timer_start_train
            }
            for key_ret, dict_ret in zip(
                    ["v", "p"],
                    [v2p_res, p2v_res]):
                if dict_ret is None:
                    continue
                for key in utils.EVALKEYS:
                    csv_input.update([(f"{key_ret}-{key}", dict_ret[key])])
            self.metrics_writer.writerow(csv_input)
            self.metrics_fh.flush()

            # step lr_scheduler
            self.lr_scheduler.step(self.det_best_field_current)

            # save checkpoint
            if is_best and epoch < (self.epochs - 1):
                # save model state
                self.best_model_ckpt = copy.deepcopy(self.model.state_dict())
            if epoch == (self.epochs - 1):
                # save checkpoint
                torch.save(self.best_model_ckpt, self.log_dir / f"ckpt_ep{self.best_epoch}.pth")

        time_total = timer() - self.timer_start_train
        self.logger.info(
            "Training {} epochs took {:.3f}s / {:.3f}s/ep val".format(
                epoch, time_total, time_total / epoch))

    @torch.no_grad()
    def validate(self, val_loader, debug_max=-1):
        self.model.eval()
        max_step = len(val_loader)
        do_clip_ret = False
        # collect embeddings
        region_emb_list = []
        global_emb_list = []
        par_emb_list = []
        for step, data_dict in enumerate(val_loader):
            if step >= debug_max > -1:
                break
            (text_feats, text_mask, region_feats, region_mask,
             global_feats, global_mask, video_id) = unpack_data(data_dict, self.use_cuda, self.device)

            # forward pass
            (text_emb, region_emb, global_emb) = self.model.forward(text_feats, region_feats,
                                                                    global_feats, text_mask,
                                                                    region_mask, global_mask)

            loss1 = self.criterion(global_emb, text_emb, self.device)
            loss2 = self.criterion(region_emb, text_emb, self.device)
            loss = (0.5 * loss1) + (0.5 * loss2)

            region_emb_list.extend(region_emb.detach().cpu())
            global_emb_list.extend(global_emb.detach().cpu())
            par_emb_list.extend(text_emb.detach().cpu())

            # logging
            if step % 10 == 0:
                self.logger.info(f"Val [{step}/{max_step}] Loss {loss.item():.4f}")

        region_emb_list = torch.stack(region_emb_list, 0)
        global_emb_list = torch.stack(global_emb_list, 0)
        par_emb_list = torch.stack(par_emb_list, 0)

        # video text retrieval
        region_emb_list = F.normalize(region_emb_list).numpy()
        global_emb_list = F.normalize(global_emb_list).numpy()
        par_emb_list = F.normalize(par_emb_list).numpy()
        v2p_res, v2p_top1, v2p_ranks = utils.compute_retr_vid_to_par(
            region_emb_list, global_emb_list, par_emb_list)
        p2v_res, p2v_top1, p2v_ranks = utils.compute_retr_par_to_vid(
            region_emb_list, global_emb_list, par_emb_list)
        sum_at_1 = v2p_res["r10"] + p2v_res["r10"]
        self.logger.info(utils.EVALHEADER)
        self.logger.info(utils.retrieval_results_to_str(p2v_res, "Par2Vid"))
        self.logger.info(utils.retrieval_results_to_str(v2p_res, "Vid2Par"))
        self.logger.info(f"Retrieval done: {self.log_dir} "
                         f"{len(global_emb_list)} Items.")
        if not do_clip_ret:
            return (v2p_res, p2v_res, sum_at_1), None
