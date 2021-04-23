## Introduction
This repo provides the code for paper "Training Image Classifiers using Semi-Weak Labels". This is the first attempt to introduce semi-weak labels and we aim to motivate more people to research in this field and then ultimately advance the area of weakly-supervised machine learning.

Contacts: `Anxiang Zhang (adamzhang1679@gmail.com), Ankit Shah (aps1@andrew.cmu.edu)`

## Citation

```
@article{zhang2021training,
  title={Training image classifiers using Semi-Weak Label Data},
  author={Zhang, Anxiang and Shah, Ankit and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2103.10608},
  year={2021}
}

```

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Data Preprocessing
- First you need to download the CIFAR10 dataset at `./data` folder as this would be used in the data generation process. 
```
sh download_cifar10.sh
```

- Then we can generate the synthetic bags of data as you want. We supports different distribution of instances within a bag including `exponential`, `Poisson` and `uniform` 
  - For `exponential` and `Poisson`, you can generate via 
  ```python
   python ./meta/generate_meta.py -ntr $n_train -nte $n_test -bs $bagsize --trial=$trial --beta $beta --reuse $reuse --distribution $distribution 
   ```
    where  `$n_train` and `n_test` is the number of bags generated for training and testing, `$bagsize` is the bag size, `$trial` is different random seed setting, `$beta` is parameter of the `exponential distribution` and `$reuse` is the maximum number of times one image could appear in different bags and `$distribution` is either 'poisson' or 'exponential' 
  
   - For `uniform` distribution, run 
   ```
   python ./meta/generate_meta_uniform.py -h 
   ``` 
   for detail information
- The naming of the meta folder follows particular format.
  - For 'poisson' distribution, the formats "reuse={}.n={}.bag={}.beta={}.poisson".format(reuse, n_training_bags, bag_size, beta)
  - For other distribution, the naming is similar with some minor differences. Please check the source code for details.
   
## Fully-Supervised Setting
For fully supervised setting, run 
```
python fully-supervised-upperbound.py --comment "" -bs $batch_size -e $n_epoch --lr $learning_rate --base_classifier "Resnet18" --meta-path=$metadata_path
```
where `$metadata_path` is the path to the dataset parameter setting such as `./meta/reuse=2.n=10000.bag=8.beta=1.2.poisson`

## Semi-Supervised Setting
For Semi-Supervised Setting, run `./semi_weak_main.py`. the parameters are as followed.
- `comment`: any string comment for this run. Used for logging purpose.
- `meta-path`: the dataset parameter setting path. e.g. `./meta/reuse=2.n=10000.bag=8.beta=1.2.poisson`
- `bs`: batch size
- `e`: maximum number of epoches.
- `t`: the trial number. For experiment, we ran 5 trials and reported the average.
- `loss-type`: the type of the regression loss. supports: KL, poisson, smoothl1, l1, l2
- `binary-loss-weight`: the weight of the classification loss or the bce loss.
- `entropy-loss-weight`: ignore this term.
- `entropy-weighted`: ignore this term.
- `regularize`: the weight of the L1 regularizer.
- `count-loss-weight`: the weight of the counting loss or the regression loss.
- `bag-size`: the bag size.
- `lr`: learning rate. default 0.1.
- `base_classifier`: the backbone feature extractor.

## Weakly-Supervised Setting
For Weakly-Supervised Setting, run `./semi_weak_main.py`. Set the `$count-loss-weight` as zero and other things equal.

## LLP Setting
For Learning-from-proportion Setting, run `./semi_weak_main.py`. Set the `$loss-type` as `KL` and other things equal.




