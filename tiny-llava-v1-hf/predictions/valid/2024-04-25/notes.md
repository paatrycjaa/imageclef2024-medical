# Notes

## Dataset

MAGICDataset output:

```
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/11k2tzj.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/121olx7.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/11hkjwl.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/11q2v3y.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/11nr7q9.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/12656ip.png
WARNING:root:Couldn't find path /content/drive/MyDrive/reddit/images/valid/11jnh7e.png
```

## Model

Input

```
model_id = 'bczhou/tiny-llava-v1-hf'
model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
print(processor)
```

Output

```
LlavaProcessor:
- image_processor: CLIPImageProcessor {
  "_valid_processor_keys": [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_center_crop",
    "crop_size",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
    "return_tensors",
    "data_format",
    "input_data_format"
  ],
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "processor_class": "LlavaProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 336
  }
}

- tokenizer: LlamaTokenizerFast(name_or_path='bczhou/tiny-llava-v1-hf', vocab_size=32000, model_max_length=2048, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
	0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32000: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32001: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}

{
  "processor_class": "LlavaProcessor"
}
```