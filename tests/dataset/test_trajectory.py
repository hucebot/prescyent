import torch

from prescyent.dataset import Trajectory
from prescyent.dataset.features import Features, Any

from tests.custom_test_case import CustomTestCase


class InitTrajectoryTest(CustomTestCase):
    def test_init__with_default_values(self):
        traj = Trajectory(
            tensor=torch.rand(100, 2, 3),
            frequency=10,
        )
        self.assertEqual(traj.duration, 10)
        self.assertEqual(traj.context, {})
        self.assertEqual(traj.tensor_features, Features([Any(range(3))]))
        self.assertEqual(traj.point_parents, [-1, -1])
        self.assertEqual(traj.point_names, ["point_0", "point_1"])
        self.assertEqual(traj.context_len, 0)
        self.assertEqual(traj.context_dims, 0)

    def test_init__with_context(self):
        traj = Trajectory(
            tensor=torch.rand(100, 2, 3),
            frequency=10,
            context={"label": torch.rand(100, 1), "image": torch.rand(100, 128)},
        )
        self.assertEqual(traj.context_len, 2)
        self.assertEqual(traj.context_dims, 129)

    def test_init_bad_args(self):
        with self.assertRaises(AssertionError):
            # the sum of Feature ids in Features must be same len as dim 2 of tensor
            Trajectory(
                tensor=torch.rand(100, 2, 3),
                frequency=10,
                tensor_features=Features([Any(range(2))]),
            )
        with self.assertRaises(AssertionError):
            # point_parents must be same len as dim 1 of tensor
            Trajectory(tensor=torch.rand(100, 2, 3), frequency=10, point_parents=[-1])
        with self.assertRaises(AssertionError):
            # point_parents must be existing parents ids
            Trajectory(
                tensor=torch.rand(100, 2, 3), frequency=10, point_parents=[-1, 2]
            )
        with self.assertRaises(AssertionError):
            # point_names must be same len as dim 1 of tensor
            Trajectory(
                tensor=torch.rand(100, 2, 3),
                frequency=10,
                point_names=["a", "b", "c", "d"],
            )
        with self.assertRaises(AssertionError):
            # context must have same freq as tensor => dim 0 of all items of == dim 0 of tensor
            Trajectory(
                tensor=torch.rand(100, 2, 3),
                frequency=10,
                context={"a": torch.rand(101, 4)},
            )
        with self.assertRaises(AssertionError):
            # context must have same freq as tensor => dim 0 of all items of == dim 0 of tensor
            Trajectory(
                tensor=torch.rand(100, 2, 3),
                frequency=10,
                context={"a": torch.rand(100, 4), "b": torch.rand(50, 4)},
            )
        # This context is ok !
        Trajectory(
            tensor=torch.rand(100, 2, 3),
            frequency=10,
            context={"a": torch.rand(100, 4), "b": torch.rand(100, 4)},
        )


class FrequencyTrajectoryTest(CustomTestCase):
    def test_freq_with_context(self):
        traj = Trajectory(
            tensor=torch.rand(100, 2, 3),
            frequency=10,
            context={"a": torch.rand(100, 4)},
        )
        traj.update_frequency(5)  # Downsample
        self.assertEqual(traj.duration, 10)  # Still 10 seconds
        self.assertEqual(traj.tensor.shape[0], 50)
        self.assertEqual(traj.context["a"].shape[0], 50)
        with self.assertRaises(AttributeError):
            traj.update_frequency(
                10
            )  # Cannot upsample a traj with context ! We don't know what it is, how to upsample it ?
