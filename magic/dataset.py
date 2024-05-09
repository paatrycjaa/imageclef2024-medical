import json
import logging
import os

from PIL import Image
from torch.utils.data import Dataset


class MAGICDataset(Dataset):
    """_summary_

    :param _type_ Dataset: MAGICDataset for ImageCLEF 2024 challenge
    """

    def __init__(self, file_path: str = "data/", split: str = "train"):
        """
        :param split: which dataset should be chosen
        :param file_path: main path with data
        """
        self.json_file = file_path + split + "_downloaded.json"
        self.folder_path = file_path + "images/" + split
        self.data = self._get_preprocessed_data()

    def _get_preprocessed_data(self):
        with open(self.json_file, encoding="utf8") as f:
            json_data = json.load(f)
        temp_data = []
        for sample in json_data:
            if len(sample["image_ids"]) != 1:
                logging.warning(
                    f'Different number of images ({len(sample["image_ids"])}) for question than 1'
                )
            image_path = self.folder_path + "/" + sample["image_ids"][0] + ".jpg"
            if not os.path.exists(image_path):
                image_path = self.folder_path + "/" + sample["image_ids"][0] + ".png"
                if not os.path.exists(image_path):
                    logging.warning(f"Couldn't find path {image_path}")
                    continue
            query_content_en = (
                ""
                if sample["query_content_en"] in ["[removed]", "[deleted]"]
                else sample["query_content_en"]
            )
            temp_data.append(
                {
                    "image": image_path,
                    "description": sample["query_title_en"] + ";" + query_content_en,
                    # "answer": sample["responses"][0]["content_en"], # COMMENTED ONLY FOR TEST SET
                    "encounter_id": sample["encounter_id"],
                }
            )
        return temp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = (
            "This is additional information about the dermatology issue on the image: "
            + sample["description"]
            + " What dermatological disease is on the image and how can it be treated?"
        )
        # prompt = (
        #     "Patient wants to find out what dermatological disease he suffers from. Considering patient additional description: "
        #     + sample["description"] +
        #     " Answer two questions: 1. What dermatological disease is on the image? 2. How can it be treated?"
        # )
        return {
            "image": Image.open(sample["image"]),  # Should be a PIL image
            "qa": [
                {
                    "question": prompt,
                    # "answer": sample["answer"],  # COMMENTED ONLY FOR TEST SET
                }
            ],  ## Why array?
            "encounter_id": sample["encounter_id"],
        }


if __name__ == "__main__":

    dataset = MAGICDataset(split="train")

    for sample in dataset:
        print(sample)
        break
