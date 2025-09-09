# ReBaNO: Reduced Basis Neural Operator Mitigating Generalization Gaps and Achieving Discretization Invariance

### Haolan Zheng<sup>1</sup>, Yanlai Chen<sup>1</sup>, Jiequn Han<sup>2</sup>, Yue Yu<sup>3</sup>

## Paper Link: [arXiv]()

![Image 1](figs/rebano_schematics.png)
*ReBaNO Architecture*

## Abstract:
<em>We propose a novel data-lean operator learning algorithm, the Reduced Basis Neural Operator (ReBaNO), to solve a group of PDEs with multiple distinct inputs. Inspired by the Reduced Basis Method and the recently introduced Generative Pre-Trained Physics-Informed Neural Networks, ReBaNO relies on a mathematically rigorous greedy algorithm to build its network structure offline adaptively from the ground up. Knowledge distillation via task-specific activation function allows ReBaNO to have a compact architecture requiring minimal computational cost online while embedding physics. In comparison to state-of-the-art operator learning algorithms such as PCA-Net, DeepONet, FNO, and CNO, numerical results demonstrate that ReBaNO significantly outperforms them in terms of eliminating/shrinking the generalization gap for both in- and out-of-distribution tests and being the only operator learning algorithm achieving strict discretization invariance. 


</sub></sub><sub>1</sup> Department of Mathematics, University of Massachusetts Dartmouth, North Dartmouth, MA.</sub></sub><br>
</sub></sub><sub>2</sup> Flatiron Institute, New York, NY.</sub></sub><br>
</sub></sub><sub>3</sup> Department of Mathematics, Lehigh University, Bethlehem, PA.</sub></sub><br>

## Requirements:
```
Python  = 3.10.14
NumPy   = 1.26.4
PyTorch = 2.4.0+cu124
```


