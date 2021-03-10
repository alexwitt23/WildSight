import io
import cv2
import pathlib
import torch
import flask
import numpy as np
import albumentations as alb
from flask import Flask, jsonify, request, render_template
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw

from wild_sight.core import detector

app = flask.Flask(__name__, template_folder=pathlib.Path(__file__).parent / "templates")


model = detector.Detector(timestamp="2021-03-06T18.30.03")
model.eval()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.get("file")
        if not file:
            return
        img_bytes = file.read()
        image_ori = Image.open(io.BytesIO(img_bytes))
        augs = alb.Compose([alb.Resize(512, 512), alb.Normalize()])
        img = torch.Tensor(augs(image=np.asarray(image_ori))["image"])
        img = img.permute(2, 0, 1).unsqueeze(0)
        img_w, img_h = image_ori.size

        boxes = model(img)
        draw = ImageDraw.Draw(image_ori)
        for box in boxes[0]:
            bbox = box.box
            bbox *= torch.Tensor([img_w, img_h, img_w, img_h])
            bbox = bbox.int().tolist()
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2
            )

        image_ori.save("/tmp/img.jpg")

        return render_template("result.html", image=image_ori)
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
