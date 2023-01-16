from typing import List
from enum import Enum


class Mode(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class Episodes():
    train: list
    test: list
    val: list
    mode: Mode

    def __init__(self, train: List, test: List, val: List) -> None:
        self.train = train
        self.test = test
        self.val = val
        self.mode = Mode.TEST

    def __getitem__(self, index):
        return getattr(self, self.mode)[index], getattr(self, self.mode)[index]

    def __len__(self):
        return len(getattr(self, self.mode))

    def _all_len(self):
        return len(self.train) + len(self.test) + len(self.val)
