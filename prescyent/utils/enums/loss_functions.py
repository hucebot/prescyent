from enum import Enum


class LossFunctions(str, Enum):
    L1LOSS = "l1loss"
    MSELOSS = "mseloss"
    NLLLOSS = "nllloss"
    CROSSENTROPYLOSS = "crossentropyloss"
    HINGEEMBEDDINGLOSS = "hingeembeddingloss"
    MARGINRANKINGLOSS = "marginrankingloss"
    TRIPLETMARGINLOSS = "tripletmarginloss"
    KLDIVLOSS = "kldivloss"
    MPJPELOSS = "mpjpeloss"
    POSITION3DLOSS = "position3dloss"
