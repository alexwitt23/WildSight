"""This file ensures the offered classification models are callable and produce
the expected ouput tensor sizes.

Please add to this test each new classifier that is supported. This models
are present in the classifier.load_backbone member function."""

import torch

import unittest

from hawk_eye.core import classifier


class ClassifierModels(unittest.TestCase):
    @torch.no_grad()
    def _test_model_output(
        self, model: classifier.Classifier, num_classes: int
    ) -> bool:
        image = torch.randn(1, 3, 64, 64)
        predictions = model.classify(image, probability=True)
        return predictions.shape == torch.Size([1, num_classes])

    def test_vovnet_19_clf_dw(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="vovnet-19-clf-dw")
        self.assertTrue(self._test_model_output(model, 2))

    def test_vovnet_19_slim_dw(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="vovnet-19-slim-dw")
        self.assertTrue(self._test_model_output(model, 2))

    def test_vovnet_19_dw(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="vovnet-19-dw")
        self.assertTrue(self._test_model_output(model, 2))

    def test_vovnet_19_slim(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="vovnet-19-slim")
        self.assertTrue(self._test_model_output(model, 2))

    def test_vovnet_39(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="vovnet-39")
        self.assertTrue(self._test_model_output(model, 2))

    def test_rexnet_v1(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="rexnet-v1")
        self.assertTrue(self._test_model_output(model, 2))

    def test_rexnet_lite0(self) -> None:
        model = classifier.Classifier(num_classes=2, backbone="rexnet-lite0")
        self.assertTrue(self._test_model_output(model, 2))


if __name__ == "__main__":
    unittest.main()
