

import pathlib
import json

import cv2


save_dir = pathlib.Path("/tmp/viz")
save_dir.mkdir(exist_ok=True, parents=True)

data_dir = pathlib.Path("/home/alex/datasets/swift-parrots/dataset-without-tails-new/images").expanduser()
data_path = pathlib.Path("/home/alex/datasets/swift-parrots/dataset-without-tails-new/annotations.json").expanduser()
data = json.loads(data_path.read_text())


images = {}
for image in data["images"]:
    images[image["id"]] = {
        "file_name": data_dir / image["file_name"],
        "annotations": [],
    }

for anno in data["annotations"]:
    if anno["image_id"] in images:
        images[anno["image_id"]]["annotations"].append(anno)
ids_map = {idx: img_id for idx, img_id in enumerate(images.keys())}


for image, labels in images.items():
    print(image, labels)

    image_arr = cv2.imread(str(labels["file_name"]))

    for label in labels["annotations"]:
        box = label["bbox"]
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image_arr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    save_path = save_dir / pathlib.Path(labels["file_name"]).name

    cv2.imwrite(str(save_path), image_arr)
