# segregate-relate-imagine

A minimal implementation of SRI on top of GENESIS-v2 (MoG likelihood). The SRI model code can be found in `./sri/model.py`.
Currently only supports training on ShapeStacks, with other datasets and evaluation code coming soon.

## Global Latent Variable Random Walks

![ShapeStacks](./videos/SRI_shapestacks_random_walk.mp4) ![ObjectsRoom](./videos/SRI_objects_room_random_walk.mp4) ![CLEVR6](./videos/SRI_clevr6_random_walk.mp4)

## Installation

With conda and Python 3.8
```shell
conda create -n sri python=3.8
pip install -r requirements.txt
```

## ShapeStacks

You need about 30GB of free disk space for [ShapeStacks](https://ogroth.github.io/shapestacks/):

```shell
# Download and extract compressed dataset
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/ShapeStacks-Manual.md  # ShapeStacks-Manual.md
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz  # shapestacks-meta.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz  # shapestacks-rgb.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-iseg.tar.gz  # shapestacks-iseg.tar.gz
```

## Training

```shell
python3 -m torch.distributed.run --nproc_per_node=1 --rdzv_endpoint='127.0.0.1':29274 train.py --DDP_port=29274 --out_dir=$OUT_DIR --data_dir=$DATA_DIR --batch_size=32 --seed=42 --run_suffix='sri_shapestacks' --tqdm
```

## Computing FID score

```shell
python3 -m torch.distributed.run --nproc_per_node=1 --rdzv_endpoint='127.0.0.1':29750 compute_fid.py --DDP_port=29750 --checkpoint_dir=$RESULTS_DIR --data_dir=$DATA_DIR --checkpoint=$CHECKPOINT --seed=1
```

## Model weights

We provide trained weights for SRI-MoG trained on ShapeStacks at `./model/sri_shapestacks`. It achieves an FID score of ~68. 

## Citation

TBD