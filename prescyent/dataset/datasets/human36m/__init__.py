"""
Module for the Human3.6m dataset
`Ionescu C, Papava D, Olaru V, Sminchisescu C (2014)
Human3.6m: Large scale datasets and predictive methods
for 3d human sensing in natural environments
TPAMI 36(7):1325–1339`
It as been implemented following examples of https://github.com/dulucas/siMLPe/
with default values intended to reproduce benchmarks like
`History Repeats Itself: Human Motion Prediction via Motion Attention
Wei, Mao and Miaomiao, Liu and Mathieu, Salzemann
ECCV 2020`
or
`Back to MLP: A Simple Baseline for Human Motion Prediction
Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier,
Alameda-Pineda and Francesc, Moreno-Noguer
arXiv preprint arXiv:2207.01567 2022`
"""

from .dataset import H36MDataset
from .config import H36MDatasetConfig
