<img src="./vit.gif"></img>

# ViT
Efficient, scalable training implementation of Google's Vision Transformer, on Cifar10 and ImageNet, in PyTorch with Weights and Biases and BentoML integration.

## Acknowledgement:
I have been greatly inspired by the brilliant code of [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

## Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

## Usage:
```bash
$ cd ViT
$ cd vit
$ colossalai run --nproc_per_node 1 train.py --use_trainer
```

## TODO:
- [x] Add logging with Weights and Biases
- [x] Build data loaders for CIFAR10 and ImageNet
- [ ] Add serving, deployment, and inference with BentoML
- [x] Implement FP16 training
- [x] Implement ZeRO
- [x] Implement Data Parallelism
- [x] Implement Tensor Parallelism

## Author:
- Enrico Shippole

## Citations:

```bibtex
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```