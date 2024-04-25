
from PIL import Image

import requests
import torch
from finetune.dataset import MAGICDataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
#from llava.model import LlavaLlamaForCausalLM
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import logging
import os
if __name__ == "__main__" :

    print("START")
    model_id = 'bczhou/tiny-llava-v1-hf'
    dataset = MAGICDataset("/content/drive/MyDrive/reddit/", "valid")

    model = LlavaForConditionalGeneration.from_pretrained(model_id)

    processor = AutoProcessor.from_pretrained(model_id)
   

    response = []
    for sample in dataset:
        text = f"USER: <image>\n {sample['qa'][0]['question']} ASSISTANT:"
        inputs = processor(text, sample['image'], return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        decoded_response = processor.decode(output[0][2:], skip_special_tokens=True)
        print(decoded_response)
        result = {
            "encounter_id": sample["encounter_id"],
            "responses": [{
                "content_en": decoded_response.split("ASSISTANT:")[1]
            }]
        }
        response.append(result)
        with open('/content/drive/MyDrive/reddit/valid_data.json', 'w') as f:
          json.dump(response, f)

