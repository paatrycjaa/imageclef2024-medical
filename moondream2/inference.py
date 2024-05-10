from magic.dataset import MAGICDataset

from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def print_sample_info(encounter_id, question, answer):
    print("-------------")
    print(f"{encounter_id=}")
    print(f"{question=}")
    print(f"{answer=}")
    print("-------------")


if __name__ == "__main__":
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    print("START")
    dataset = MAGICDataset("/content/drive/MyDrive/reddit/", "valid")

    response = []
    for sample in dataset:
        encounter_id = sample["encounter_id"]
        question = sample['qa'][0]['question']
        encoded_image = model.encode_image(sample['image'])
        answer = model.answer_question(encoded_image, question, tokenizer)
        result = {
            "encounter_id": encounter_id,
            "responses": [{
                "content_en": answer
            }]
        }
        response.append(result)
        with open('/content/drive/MyDrive/reddit/2025-05-02-valid-prediction.json', 'w') as f:
            json.dump(response, f)
        print_sample_info(encounter_id, question, answer)
