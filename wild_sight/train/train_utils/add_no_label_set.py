"""Take in an COCO annotation json and add to it a folder
of images that do not have any of the relavant classes."""


import json
import pathlib
import shutil

json_file = pathlib.Path("/home/alex/datasets/whale-giraffe-zebra-coco/annotations-copy.json")
data = json.loads(json_file.read_text())

image_dir = pathlib.Path("/home/alex/Downloads/archive (2)/shark-images")

save_dir = pathlib.Path("/home/alex/datasets/whale-giraffe-zebra-coco/images") / image_dir.name
save_dir.mkdir(exist_ok=True, parents=True)

for image in image_dir.glob("*.jpg"):

    shutil.copy2(image, save_dir / image.name)
    data["images"].append({"file_name": f"{save_dir.name}/{image.name}", "id": len(data["images"])})
    
json_file.write_text(json.dumps(data))