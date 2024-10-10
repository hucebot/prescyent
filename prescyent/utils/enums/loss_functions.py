from enum import Enum


class LossFunctions(str, Enum):
    """Map to the required loss function in the torch_module class"""

    L1LOSS = "l1loss"
    """torch.nn.L1Loss"""
    MSELOSS = "mseloss"
    """torch.nn.MSELoss"""
    NLLLOSS = "nllloss"
    """torch.nn.NLLLoss"""
    CROSSENTROPYLOSS = "crossentropyloss"
    """torch.nn.CrossEntropyLoss"""
    HINGEEMBEDDINGLOSS = "hingeembeddingloss"
    """torch.nn.HingeEmbeddingLoss"""
    MARGINRANKINGLOSS = "marginrankingloss"
    """torch.nn.MarginRankingLoss"""
    TRIPLETMARGINLOSS = "tripletmarginloss"
    """torch.nn.TripletMarginLoss"""
    KLDIVLOSS = "kldivloss"
    """torch.nn.KLDivLoss"""
    MFRDLOSS = "mfdloss"
    """MeanFinalRigidDistanceLoss"""
    MTRDLOSS = "mtdloss"
    """MeanTotalRigidDistanceLoss"""
    MTRDVLOSS = "mtrdvloss"
    """MeanTotalRigidDistanceAndVelocityLoss"""
