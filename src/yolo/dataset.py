import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset

from pathlib import Path


class YOLOVocDataset(Dataset):
    def __init__(self, root: Path | str, year: str, image_set: str, S=7, B=2, C=20):
        self._S = S
        self._B = B
        self._C = C

        self._voc_dataset = VOCDetection(
            year=year,
            root=root,
            image_set=image_set,
        )

        self._transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor()
        ])

        self._class_map = {
            class_label: i for i, class_label in enumerate([
                "aeroplane",
                "bicycle",
                "bird",
                "boat",
                "bottle",
                "bus",
                "car",
                "cat",
                "chair",
                "cow",
                "diningtable",
                "dog",
                "horse",
                "motorbike",
                "person",
                "pottedplant",
                "sheep",
                "sofa",
                "train",
                "tvmonitor",
            ])
        }

    def __len__(self):
        return len(self._voc_dataset)

    def __getitem__(self, index: int):
        """
        Outputs a tuple of the image as a tensor and the 7 x 7 x 30 vector expected by YOLO.

        :param index: The index to get
        :type index: int
        """
        pil_image, annotations = self._voc_dataset[index]

        print(annotations)

        w_orig = int(annotations["annotation"]["size"]["width"])
        h_orig = int(annotations["annotation"]["size"]["height"])

        label_matrix = torch.zeros((self._S, self._S, 5 + self._C))

        objs = annotations["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            class_index = self._class_map[obj["name"]]
            bounding_box = obj["bndbox"]

            x1, y1 = float(bounding_box['xmin']), float(bounding_box['ymin'])
            x2, y2 = float(bounding_box['xmax']), float(bounding_box['ymax'])

            # Normalize coordinates
            x_center = ((x1 + x2) / 2) / w_orig
            y_center = ((y1 + y2) / 2) / h_orig
            width = (x2 - x1) / w_orig
            height = (y2 - y1) / h_orig

            i, j = int(y_center * self._S), int(x_center * self._S)

            # If the cell does not have an object, assign it this object
            if label_matrix[i, j, 4] == 0:
                label_matrix[i, j, 4] = 1

                # Make coordinates relative to the cell
                x_cell = self._S * x_center - j
                y_cell = self._S * y_center - i

                # Fill the 30-dimensional vector associated with the grid cell
                label_matrix[i, j, 0:4] = torch.tensor(
                    [x_cell, y_cell, width, height])
                label_matrix[i, j, 5 + class_index] = 1

        return self._transform(pil_image), label_matrix
