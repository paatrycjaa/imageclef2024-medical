import json
import logging
import os

from PIL import Image
from torch.utils.data import Dataset

class MAGICDataset(Dataset):
    """_summary_

    :param _type_ Dataset: MAGICDataset for ImageCLEF 2024 challenge
    """
    def __init__(self, split="train"):
        self.json_file = "data/" + split + "_downloaded.json"
        self.folder_path = "data/" + split
        self.data = self._get_preprocessed_data()

    def _get_preprocessed_data(self):
        with open(self.json_file, encoding="utf8") as f :
            json_data = json.load(f)
        temp_data = []
        for sample in json_data:
            if len(sample["image_ids"]) != 1 :
                logging.warning(f'Different number of images ({len(sample["image_ids"])}) for question than 1')
            image_path = self.folder_path + '/' + sample["image_ids"][0] + '.jpg'
            if not os.path.exists(image_path):
                image_path = self.folder_path + '/' + sample["image_ids"][0] + '.png'
                if not os.path.exists(image_path):
                    logging.warning(f"Couldn't find path {image_path}")
                    continue
            temp_data.append({
                "image" : image_path,
                "description" : sample["query_title_en"],
                "answer" : sample["responses"][0]["content_en"]
            })
        return temp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = (
            "This is additional information about the dermatology issue on the image:"
            + sample["description"]
            + "What dermatological disease is on the image and how can it be treated?"
        )
        return {
            "image": Image.open(sample["image"]),  # Should be a PIL image
            "qa": [
                {
                    "question": prompt,
                    "answer": sample["answer"],
                }
            ],
        }

if __name__ == "__main__" :

    dataset = MAGICDataset("train")

    for sample in dataset:
        print(sample)
        break
