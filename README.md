# Road Planning for Slums
![Loading Model Overview](assets/overview.png "Model Overview")
---

In this project, we echo the *AI for Good* initiative and propose a deep reinforcement learning approach to perform road planning for slums, a critical issue in urban sustainability.
Our proposed model can generate road plans that significantly improve the connectivity of slums at minimal costs.
Experiments on slums from different countries demonstrates a relative improvement of over 14.3% in accessibility compared with existing baselines.

This project was initially described in the full paper [Road Planning for Slums via Deep Reinforcement Learning](https://arxiv.org/abs/2305.13060) at [KDD 2023](https://kdd.org/kdd2023/) in Long Beach, CA. 
Contributors to this project are from the [Future Intelligence laB (FIB)](https://fi.ee.tsinghua.edu.cn/) at [Tsinghua University](https://www.tsinghua.edu.cn/en/).


## Installation 

### Environment
* Tested OS: Linux
* Python >= 3.8
* PyTorch == 1.10.1
* Tensorboard
### Dependencies:
1. Install [PyTorch 1.10.1](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Set the following environment variable to avoid problems with multiprocess trajectory sampling:
```
export OMP_NUM_THREADS=1
```




## Training

You can train your own models using the provided config in `road_planning/cfg`:

```
python -m road_planning.train --slum_name capetown1 --cfg demo --global_seed 0
```
You can replace `demo` to train other cfgs.

The results are saved in `train_data/slum_name/cfg/seed`

### Convergence
Our models, both DRL-GNN and DRL-MLP, usually converges with less than 100 iterations, which takes about 2 hours on a single server with an Nvidia
GeForce 2080Ti GPU.
We show the episodic reward (normalized to 0-1) after each iteration of training in the following figures.
|Harare, ZWE|Cape Town, ZAF|
|--|--|
| ![Loading convergence curve](assets/iteration_reward_harare.png "Convergence curve on Harare, ZWE") | ![Loading convergence curve](assets/iteration_reward_capetown.png "Convergence curve on Cape Town, ZAF") |

## Results
- For accessibility, it is desired to achieve universal connectivity as early as possible, thus we calculate the **n**umber of **r**oad segments (**NR**) consumed to achieve universal connectivity.
- For travel distance, we compute the **a**verage **d**istance (**AD**) between any pair of places in the slum over the road network.
- We define the construction cost of each road segment as its length, and calculate the **s**um of **c**osts (**SC**) of all planned roads.

It is worth noting that all the metrics are **the lower the better**.

![Loading Overall Performance](assets/overall_results.png "Overall Results")

## Citation
If you use this code in your project, please consider citing the following paper:
```bibtex
@inproceedings{zheng2023road,
    author = {Zheng, Yu and Su, Hongyuan and Ding, Jingtao and Jin, Depeng and Li, Yong},
    title = {Road Planning for Slums via Deep Reinforcement Learning},
    year = {2023},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {5695–5706},
}
```

## License

The software in this repo is freely available under MIT license. Please see the [license](LICENSE) for further details.

## Note

The implemention is based on *[Transform2Act](https://github.com/Khrylx/Transform2Act)*.
