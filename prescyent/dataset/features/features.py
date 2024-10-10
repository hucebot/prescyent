"""list of features, giving unique name to each feature"""
from typing import List

from .feature import Feature


class Features(list):
    """class representing a list of features
    by default we give a unique name to each feature, ordering them and using their index
    """

    def __init__(self, features: List[Feature], index_name: bool = True) -> None:
        features.sort(key=lambda f: f.ids[0])
        for f, feat in enumerate(features):
            if index_name:
                feat.name = f"{feat.name}_{f}"
            self.append(feat)

    @property
    def ids(self):
        ids = []
        for feat in self:
            ids += feat.ids
        return ids

    def __hash__(self):
        return hash(str(self))
