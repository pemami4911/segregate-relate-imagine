# segregate-relate-imagine

A minimal implementation of SRI on top of GENESIS-v2 (MoG likelihood). The SRI mode code can be found in `./sri/model.py`.

## Installation

With conda and Python 3.8
```shell
conda create -n sri python=3.8
pip install -r requirements.txt
```

## ShapeStacks

You need about 30GB of free disk space for [ShapeStacks](https://ogroth.github.io/shapestacks/):

```shell
# Download compressed dataset
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/ShapeStacks-Manual.md  # ShapeStacks-Manual.md
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz  # shapestacks-meta.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz  # shapestacks-rgb.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-iseg.tar.gz  # shapestacks-iseg.tar.gz
cd -
```

### Training

```shell
python3 -m torch.distributed.run --nproc_per_node=1 --rdzv_endpoint='127.0.0.1':29274 train.py --DDP_port=29274 --out_dir=$OUT_DIR --batch_size=32 --seed=42 --run_suffix='sri_shapestacks' --tqdm
```