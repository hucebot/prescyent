#!/usr/bin/env python3
from argparse import ArgumentParser

import rclpy

from prescyent.auto_predictor import get_predictor_from_path
from prescyent.ros2.predictor_node import PredictorNode


def main(args):
    rclpy.init(args=None)
    args.predictor = get_predictor_from_path(args.predictor)
    node = PredictorNode(predictor=args.predictor,
                         history_size=args.history_size,
                         future_size=args.future_size,
                         time_step=args.time_step)
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictor", default=None)
    parser.add_argument("--history_size", default=10)
    parser.add_argument("--future_size", default=10)
    parser.add_argument("--time_step", default=10)
    args = parser.parse_args()
    main(args)
