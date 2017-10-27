### Detector

#### To Train

```
usage: train.py [-h] --data_root DATA_ROOT [--batch_size b] [--epoch EPOCH]
                [--nworkers w] [--cuda] [--lr LR]
                [--log_interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        root path of data
  --batch_size b, -B b  batch size
  --epoch EPOCH         number of epoch
  --nworkers w          number of dataloader workers
  --cuda                use gpu
  --lr LR, --learning-rate LR
                        initial learning rate
  --log_interval LOG_INTERVAL
                        Print log every N batches
```

This script assumes you have a directory called ```pretrained``` where the pretrained model exists.

alexnet pretrained model: https://www.dropbox.com/sh/zecyw77zp74amx1/AAATgVjfJCLdyvVkDH2cmw8Ga?dl=0

---

#### Demo

```
usage: demo.py [-h] [-i INP] [-hd HD] [-s SIZE] [-t THRESHOLD] model

Person Detection Demo

positional arguments:
  model                 model path

optional arguments:
  -h, --help            show this help message and exit
  -i INP, --inp INP     camera device index, default 0
  -hd HD                process full frame or resize to net eye size only
  -s SIZE, --size SIZE  network input size
  -t THRESHOLD, --threshold THRESHOLD
                        detection threshold
```

size is ignored if hd is ```True```. size center crops and rescales. 
