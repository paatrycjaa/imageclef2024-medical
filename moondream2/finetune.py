import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm
from dataset import MAGICDataset


# from flash_attn import flash_attn_func, flash_attn_varlen_func
# from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

DEVICE = "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-04-02"

# Number of tokens used to represent each image.
IMG_TOKENS = 729
ANSWER_EOS = "<|endoftext|>"

class Finetune():

    def __init__(self, train_dataset, valid_dataset, params):
        self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
                #attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
                torch_dtype=DTYPE, device_map={"": DEVICE}
            )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.parameters = params



    def _collate_fn(self, batch):
        images = [sample['image'] for sample in batch]
        print(images)
        images = torch.stack(self.model.vision_encoder.preprocess(images))
        images = rearrange(images,
                        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                        p1=14, p2=14)

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = self.tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )
    
    def _compute_loss(self, batch):
        images, tokens, labels, attn_mask = batch

        images = images.to(DEVICE)
        tokens = tokens.to(DEVICE)
        labels = labels.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        with torch.no_grad():
            img_embs = self.model.vision_encoder.encoder(images)
            img_embs = self.model.vision_encoder.projection(img_embs)

        tok_embs = self.model.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
        )

        return outputs.loss
    
    def _lr_schedule(self, step, max_steps):
        x = step / max_steps
        if x < 0.1:
            return 0.1 * self.parameters["learning_rate"] + 0.9 * self.parameters["learning_rate"] * x / 0.1
        else:
            return 0.1 * self.parameters["learning_rate"] + 0.9 * self.parameters["learning_rate"] * (1 + math.cos(math.pi * (x - 0.1))) / 2
        
    def run(self):

        dataloaders = {
            "train": DataLoader(
                self.train_dataset,
                batch_size=self.parameters['batch_size'],
                shuffle=True,
                collate_fn=self._collate_fn,
            ),
            "val": DataLoader(
                self.valid_dataset,
                batch_size=self.parameters['batch_size'],
                collate_fn=self._collate_fn,
            ),
        }

        self.model.text_model.train()
        self.model.text_model.transformer.gradient_checkpointing_enable()

        total_steps = self.parameters['epochs'] * len(dataloaders["train"]) // self.parameters['grad_accum_steps']
        optimizer = Adam8bit(
            [
                {"params": self.model.text_model.parameters()},
            ],
            lr=self.parameters["learning_rate"] * 0.1,
            betas=(0.9, 0.95),
            eps=1e-6
        )

        i = 0
        for epoch in range(self.parameters['epochs']):
            for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{self.parameters['epochs']}"):
                i += 1

                loss = self._compute_loss(batch)
                loss.backward()

                if i % self.parameters['grad_accum_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    lr = self._lr_schedule(i / self.parameters['grad_accum_steps'], total_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                # if i % 100 == 0 and USE_WANDB:
                #     # Calculate validation loss
                #     val_loss = 0
                #     for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                #         with torch.no_grad():
                #             val_loss += compute_loss(val_batch).item()
                #     val_loss /= len(dataloaders["val"])

                # if USE_WANDB:
                #     wandb.log({
                #         "loss/train": loss.item(),
                #         "lr": optimizer.param_groups[0]['lr']
                #     } | ({"loss/val": val_loss} if i % 100 == 0 else {}))
        # if USE_WANDB:
        #     wandb.finish()
        self.model.save_pretrained("checkpoints/moondream-ft")



if __name__ == "__main__" :

    parmas = {
        'grad_accum_steps' : 1,
        'batch_size' : 8,
        'epochs' : 2,
        'learning_rate' : 3e-5
    }

    finetune = Finetune(train_dataset=MAGICDataset('train'), valid_dataset=MAGICDataset('valid'), params= parmas)
    finetune.run()


