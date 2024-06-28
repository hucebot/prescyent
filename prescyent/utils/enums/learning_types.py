from enum import Enum


class LearningTypes(str, Enum):
    """Method used to generate the MotionDataSample in the dataloaders"""

    SEQ2SEQ = "sequence_2_sequence"
    """Generate pairs with the following behavior, with the expected in points and features and out points and features :
        - given a time step T, history_size H and future_size F we have:
        x a sequence of lenght H with the frames = [T-H, .. T]
        y a sequence of lenght F with the frames = [T+1, ... T+F]
    """
    AUTOREG = "auto_regressive"
    """Generate pairs with the following behavior, with the expected in points and features and out points and features :
        - given a time step T, history_size H and future_size F we have:
        x a sequence of lenght H with the frames = [T-H, .. T]
        y a sequence of lenght H with the frames = [T-H+1, ... T+1]
    """
    SEQ2ONE = "sequence_2_one"
    """Generate pairs with the following behavior, with the expected in points and features and out points and features :
        - given a time step T, history_size H and future_size F we have:
        x a sequence of lenght H with the frames = [T-H, .. T]
        y a sequence of lenght 1 with the frames = [T+F]
    """
