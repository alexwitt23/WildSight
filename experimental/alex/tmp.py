import pathlib
import json


fi = pathlib.Path("/home/alex/datasets/ENA-24/annotations_new.json")
new_json = pathlib.Path("/home/alex/datasets/ENA-24/annotations_newer.json")

data = json.loads(fi.read_text())
data = json.loads(new_json.read_text())
print(data["categories"])
raise ValueError
print(list(data.keys()))
print(data)
print(data["categories"])


data_dir = pathlib.Path("~/datasets/ENA-24").expanduser()
images = {}
for image in data["images"]:
    images[image["id"]] = {
        "file_name": str(data_dir / image["file_name"]),
        "annotations": [],
    }


for anno in data["annotations"]:
    if anno["image_id"] in images:
        images[anno["image_id"]]["annotations"].append(anno)

import cv2

images = {
    img: data
    for img, data in images.items()
    if pathlib.Path(data["file_name"]).is_file()
}

# Now remove the human class and shift all the classes after 8 down one
data["categories"] = [cat for cat in data["categories"] if cat["id"] != 8]
for cat in data["categories"]:
    if cat["id"] > 8:
        cat["id"] -= 1

data["annotations"] = [box for box in data["annotations"] if box["category_id"] != 8]
for box in data["annotations"]:
    if box["category_id"] > 8:
        box["category_id"] -= 1

save_dir = pathlib.Path("/tmp/imgs_dir")
save_dir.mkdir(exist_ok=True)
cat = {idx["id"]: 0 for idx in data["categories"]}
"""
for img, im_data in images.items():

    im_data["annotations"] = [box for box in im_data["annotations"] if box["category_id"] != 8]
    for box in im_data["annotations"]:
        bbox = box["bbox"]
        cat[box["category_id"]] += 1
        if box["category_id"] == 8:
            print("HUH")
        
        x, y, w, h = bbox
        cv2.rectangle(
            image,
            (int(x),int(y)),
            (int(x) + int(w),int(y) + int(h)),
            (0,255,0),
            2
        )
        
    
    #cv2.imwrite(str(save_dir / data["file_name"].name), image)
"""
print(cat)

new_json.write_text(json.dumps(data))
