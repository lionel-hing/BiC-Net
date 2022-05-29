## BiC-Net: Learning Efficient Spatio-Temporal Relation for Text-Video Retrieval
This is our implementation for the paper:

Ning Han, Jingjing Chen, Chuhao Shi, Yawen Zeng, Guangyi Xiao, and Hao Chen. 2022. BiC-Net: Learning Efficient Spatio-Temporal Relation for Text-Video Retrieval.

### Environment Settings
We use the framework pytorch.

- Python == 3.7
- Pytorch == 1.7.1
- numpy == 1.20.2


### Training

You can also follow the instruction below to train your own model.

Run train.py to  train and save models:
```
python train.py --cuda --is_train --dataset=msr-vtt --data_split=9000 --layer_num=4 --log_dir=./data/runs/xxx --dataroot=./data/MSR-VTT 
```

### Evaluation


run eval.py to evaluate models:
```
python eval.py --cuda --checkpoint= ./models/ckpt_best.pth
```
### Example to get the results

There are a lot of experimental records in the ./data/runs/xxx

### Dataset
We provide three datasets that we used in our paper: MSR-VTT, MSVD, YouCook2.
Download the processed video and text features of [MSR-VTT](https://drive.google.com/file/d/19GiZU5lqIUsdYahp5HYBECy-7hRJnkkb/view?usp=sharing), [MSVD](https://drive.google.com/file/d/1j87YAvIJ3yZ3s6tV7v8YQHMLjXxRRI3N/view?usp=sharing), [YouCook2](https://drive.google.com/file/d/1q7QocJq3mDJU0VxqJRZhSbqdtPerC4PS/view), and [YouCook2_BB](https://github.com/MichiganCOG/Video-Grounding-from-Text), and save them in `/data` folder.



Last Update Date: May 29, 2020
