"""A classifier model which wraps around a backbone. This setup allows for easy
interchangeability during experimentation and a reliable way to load saved models."""

import pathlib
from typing import Optional

import torch
import yaml

from third_party.vovnet import vovnet


class Classifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: Optional[int] = 2,
        timestamp: Optional[str] = None,
        backbone: Optional[str] = None,
        half_precision: Optional[bool] = False,
    ) -> None:
        """
        Args:
            num_classes: The number of classes to predict.
            timestamp: The timestamp of the model to download from GCloud.
            backbone: A string designating which model to load.
            half_precision: Whether to use half precision. This should be False for
                training but True during inference.

        :raises ValueError: Error if neither a timestamp or backbone arg is supplied.

        Examples:
            >>> classifier = Classifier(2, backbone="vovnet-19")
            >>> with torch.no_grad():
            ...    predictions = classifier.classify(torch.randn(1, 3, 64, 64), True)
            >>> predictions.shape
            torch.Size([1, 2])
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_cuda = torch.cuda.is_available()
        self.half_precision = half_precision and self.use_cuda

        if backbone is None and timestamp is None:
            raise ValueError("Must supply either model timestamp or backbone to load")

        # If a version is given, download it.
        if timestamp is not None:

            # For the distributed pip package, look inside `production_models`
            production_models = pathlib.Path(__file__).parent / "production_models"
            if production_models.is_dir():
                model_path = production_models / timestamp
            else:
                # Download the model or find it locally.
                model_path = asset_manager.download_model("classifier", timestamp)

            config = yaml.safe_load((model_path / "config.yaml").read_text())["model"]
            backbone = config.get("backbone", None)
            # Construct the model, then load the state
            self.model = self.load_backbone(backbone)
            self.load_state_dict(
                torch.load(model_path / "classifier.pt", map_location="cpu")
            )
            self.image_size = config["image_size"]
        else:
            # If no timestamp supplied, just load the backbone
            self.model = self.load_backbone(backbone)

        self.model.eval()

        if self.use_cuda:
            self.model.cuda()

        if self.half_precision:
            self.model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.

        Args:
            x: input tensor.

        Returns:
            the output tensor.
        """
        # If half precision, assume inference not train.
        if self.half_precision:
            x = x.half()

        return self.model(x)

    def load_backbone(self, backbone: str) -> torch.nn.Module:
        """Load the supplied backbone. See this function for the list of potential
        backbones that can be loaded.

        Args:
            backbone: The backbone type to load.

        Returns:
            The loaded model.

        :raises ValueError: If improper backbone is supplied.
        """
        if "rexnet" in backbone:
            model = rexnet.ReXNet(num_classes=self.num_classes, model_type=backbone)
        elif "vovnet" in backbone:
            model = vovnet.VoVNet(model_name=backbone, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def classify(self, x: torch.Tensor, probability: bool = False) -> torch.Tensor:
        """Take in an image batch and return the class for each image. If specified,
        softmax will be applied to the predictions.

        Args:
            x: Input tensor of size (batch, height, width, channels).
            probability: Whether or not to apply softmax.

        Returns:
            The output tensor.
        """

        if self.use_cuda and self.half_precision:
            x = x.half()
        if probability:
            return torch.nn.functional.softmax(self.model(x), dim=1)
        else:
            _, predicted = torch.max(self.model(x).data, 1)
            return predicted
