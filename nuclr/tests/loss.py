import unittest
import argparse
import torch
import torch.nn.functional as F
from ..loss import loss_by_task, metric_by_task, get_balanced_accuracy


class TestMetrics(unittest.TestCase):
    def test_loss_by_task(self):
        output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        output_map = {"a": 2, "b": 1}
        config = argparse.Namespace()
        config.TARGETS_CLASSIFICATION = ["a"]
        config.TARGETS_REGRESSION = ["b"]
        loss = loss_by_task(output, targets, output_map, config)
        self.assertEqual(loss.shape, (2,))
        self.assertEqual(loss[0], F.cross_entropy(output[:, :2], targets[:, 0].long()))
        self.assertEqual(loss[1], F.mse_loss(output[:, 2], targets[:, 1].float()))

    def test_get_balanced_accuracy(self):
        # test with class weights that are not uniform
        output = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([1.0, 1.0, 0.0])
        acc = get_balanced_accuracy(output, targets)
        self.assertEqual(acc.item(), 0.25)

    def test_metric_by_task(self):
        # test metrics with class weights that are not uniform

        output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        output_map = {"a": 2, "b": 1}
        config = argparse.Namespace()
        config.TARGETS_CLASSIFICATION = ["a"]
        config.TARGETS_REGRESSION = ["b"]
        metrics = metric_by_task(output, targets, output_map, config)
        self.assertEqual(metrics.shape, (2,))
        self.assertEqual(
            metrics[0], 100 * get_balanced_accuracy(output[:, :2], targets[:, 0].long())
        )
        self.assertEqual(
            metrics[1], F.mse_loss(output[:, 2], targets[:, 1].float()).sqrt()
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
