# Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs
This repository contains the source code for the paper [Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs](https://arxiv.org/abs/2308.10779), by [Dongjin Lee](https://github.com/wooner49), [Juho Lee](https://juho-lee.github.io/) and [Kijung Shin](https://kijungs.github.io/), presented at [AAAI 2024](https://aaai.org/aaai-conference/).

In this paper, we propose **T-SPEAR**, a simple and effective adversarial attack method for link prediction on continuous-time dynamic graphs, focusing on investigating the vulnerabilities of TGNNs. Specifically, before the training procedure of a victim model, which is a TGNN for link prediction, we inject edge perturbations to the data that are unnoticeable in terms of the four constraints we propose, and yet effective enough to cause malfunction of the victim model. Moreover, we propose a robust training approach **T-SHIELD** to mitigate the impact of adversarial attacks. By using edge filtering and enforcing temporal smoothness to node embeddings, we enhance the robustness of the victim model. Our experimental study shows that T-SPEAR significantly degrades the victim model's performance on link prediction tasks, and even more, our attacks are transferable to other TGNNs, which differ from the victim model assumed by the attacker. Moreover, we demonstrate that T-SHIELD effectively filters out adversarial edges and exhibits robustness against adversarial attacks, surpassing the link prediction performance of the naive TGNN by up to 11.2% under T-SPEAR.

## Requirements
- Python 3.9.12
- PyTorch 1.12.1
- DGL 0.9.1
- Numpy 1.22.3
- Scikit-learn 1.0.2
- PyYAML 6.0

## Reference
This code is free and open source for only academic/research purposes (non-commercial).
If you use this code as part of any published research, please acknowledge the following paper.
```
@article{lee2023spear,
  title={Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs},
  author={Lee, Dongjin and Lee, Juho and Shin, Kijung},
  journal={arXiv preprint arXiv:2308.10779},
  year={2023}
}
```
