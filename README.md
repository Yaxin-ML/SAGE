# Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning

Yaxin Hou, Jun Ma, Hanyang Li, Bo Han, Jie Yu, Yuheng Jia, Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning, International Conference on Machine Learning, 6th-11th July, Seoul, 2026.

This is an official [PyTorch](http://pytorch.org) implementation for **Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning**.

## Introduction

This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning).

What I've done is just adding our SAGE algorithm in `semilearn/imb_algorithms/sage`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run

For example, on CIFAR-10 with arbitrary unlabeled data($\gamma_u=100$)

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "/config/config-2/sage/003-fixmatch_sage_cifar10_lb40_1_ulb4996_100_random_0.0_2.yaml"
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `/saved_models/sage/003-fixmatch_sage_cifar10_lb40_1_ulb4996_100_random_0.0_2/log.txt` will tell you the best accuracy. 

For example,
```
[2025-12-21 07:10:45,980 INFO] model saved: ./saved_models/sage/003-fixmatch_sage_cifar10_lb40_1_ulb4996_100_random_0.0_2/latest_model.pth
[2025-12-21 07:10:45,990 INFO] Model result - eval/best_acc : 0.6683
[2025-12-21 07:10:45,990 INFO] Model result - eval/best_it : 133119
```

## Results

The reported accuracies in Table 1, 2, and 3 in our paper are the average over three different runs (random seeds are 0/1/2). 

## Citation

If you find our method useful, please consider citing our paper:

  ```
  @inproceedings{SAGE,
  author       = {Yaxin Hou and
                  Jun Ma and
		          Hanyang Li and
				  Bo Han and
				  Jie Yu and
			      Yuheng Jia},
  title        = {Beyond Distribution Estimation: Simplex Anchored Structural Inference
				  Towards Universal Semi-Supervised Learning},
  booktitle    = {International Conference on Machine Learning (ICML)},
  volume       = {},
  pages        = {},
  year         = {2026}
  }
  ```
