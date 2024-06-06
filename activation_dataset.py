import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

# Custom dataset class
class ActivationDataset(Dataset):
    def __init__(self, prompts, tokenizer, model, num_samples_per_prompt):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.model = model
        self.num_samples_per_prompt = num_samples_per_prompt
        self.activations = []
        self.prompt_texts = []
        self.sampled_indices = []
        self.sampled_tokens = []  # New list for decoded tokens
        self.prompt_batch_size = 8
        self.device = model.device
        self.max_length = 3000*4 # Max length of prompt

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return {
            "activations": self.activations[idx],
            "prompt_texts": self.prompt_texts[idx],
            "sampled_indices": self.sampled_indices[idx],
            "sampled_tokens": self.sampled_tokens[idx],  # Include sampled tokens in the batch
        }

    def process_prompts(self):
        for i in tqdm(range(0, len(self.prompts), self.prompt_batch_size), desc="Processing prompt batches"):
            prompt_batch = [prompt_item["text"][:self.max_length] for prompt_item in self.prompts[i : i + self.prompt_batch_size]]
            input_ids = self.tokenizer(prompt_batch, return_tensors="pt", truncation=True, padding=True).to(self.device)
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                activations = outputs.hidden_states[-1].cpu()
            
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            
            del outputs
            torch.cuda.empty_cache()
            
            for j in range(activations.size(0)):
                num_pad_tokens = input_ids[j].shape[0] - torch.sum(attention_mask[j]).item()
                valid_tokens = activations.size(1) - num_pad_tokens
                
                num_samples = min(self.num_samples_per_prompt, valid_tokens)
                sampled_indices = np.random.choice(valid_tokens, num_samples, replace=False)
                
                sampled_activations = activations[j, sampled_indices + num_pad_tokens].to(torch.float32).cpu().numpy()
                self.activations.extend(np.ascontiguousarray(sampled_activations))
                self.prompt_texts.extend([prompt_batch[j]] * num_samples)
                self.sampled_indices.extend(sampled_indices.tolist())
                
                for sampled_index in sampled_indices:
                    sampled_token = self.tokenizer.decode(input_ids[j, sampled_index + num_pad_tokens].item())
                    self.sampled_tokens.append(sampled_token)


def load_new_samples(data_iter, num_prompts_to_load):
    new_prompts = []
    for _ in range(num_prompts_to_load):
        try:
            new_prompts.append(next(data_iter))
        except StopIteration:
            break  # End of dataset
    return new_prompts