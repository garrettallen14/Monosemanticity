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
        self.prompt_batch_size = 12
        self.device = model.device

    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return {
            "activations": self.activations[idx],
            "prompt_texts": self.prompt_texts[idx],
            "sampled_indices": self.sampled_indices[idx],
        }

    def process_prompts(self):
        for i in tqdm(range(0, len(self.prompts), self.prompt_batch_size), desc="Processing prompt batches"):
            # Extract the actual prompt texts
            prompt_batch = [prompt_item["chosen"] for prompt_item in self.prompts[i : i + self.prompt_batch_size]]
            input_ids = self.tokenizer(prompt_batch, return_tensors="pt", padding=True).to(self.device)
    
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
            activations = outputs.hidden_states[-1].cpu()
            del outputs
            torch.cuda.empty_cache()
            
            for j in range(activations.size(0)):
                # 1. Get the original (unpadded) input IDs
                original_input_ids = input_ids[j][input_ids[j] != self.tokenizer.pad_token_id]  
    
                # 2. Get valid token indices (non-padded) using attention mask
                valid_token_indices = torch.where(attention_mask[j] == 1)[0]
    
                # Sample from valid indices (this is where sampled_indices is defined)
                sampled_indices = valid_token_indices[torch.randperm(valid_token_indices.size(0))[: self.num_samples_per_prompt]].cpu()
    
                # 3. Correct the sampled indices by subtracting the padding length
                num_pad_tokens = input_ids[j].shape[0] - original_input_ids.shape[0] 
                corrected_sampled_indices = sampled_indices - num_pad_tokens  
    
                # 4. Ensure indices are within the valid range
                corrected_sampled_indices = corrected_sampled_indices[
                    (corrected_sampled_indices >= 0) & (corrected_sampled_indices < original_input_ids.shape[0])
                ]
    
                # 5. Sample from the valid, corrected indices (if any remain)
                if len(corrected_sampled_indices) >= self.num_samples_per_prompt:
                    corrected_sampled_indices = corrected_sampled_indices[torch.randperm(corrected_sampled_indices.size(0))[: self.num_samples_per_prompt]].cpu()
    
                    sampled_activations = activations[j][corrected_sampled_indices].float().numpy()
                    
                    self.activations.extend(np.ascontiguousarray(sampled_activations))
                    self.prompt_texts.extend([prompt_batch[j]] * self.num_samples_per_prompt)
                    self.sampled_indices.extend(np.ascontiguousarray(corrected_sampled_indices.numpy()))  # Use the corrected indices

def load_new_samples(data_iter, num_prompts_to_load):
    new_prompts = []
    for _ in range(num_prompts_to_load):
        try:
            new_prompts.append(next(data_iter))
        except StopIteration:
            break  # End of dataset
    return new_prompts