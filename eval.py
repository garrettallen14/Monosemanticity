import torch
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm


def evaluate_sae(model, dataloader, tokenizer, num_features=5, top_activations=10, iteration=None):
    model.eval()
    device = next(model.parameters()).device

    top_activations_data = {}

    # Randomly select num_features to evaluate from (0, hidden_dim)
    feature_indices = np.random.choice(model.encoder[0].out_features, num_features, replace=False)

    # Initialize top_activations_data
    for feature_idx in feature_indices:
        top_activations_data[feature_idx] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = batch["activations"].to(device)
            prompt_texts = batch["prompt_texts"]  
            sampled_indices = batch["sampled_indices"]
            sampled_tokens = batch["sampled_tokens"]

            # Ensure these are lists for consistent indexing
            if not isinstance(prompt_texts, list):
                prompt_texts = [prompt_texts]
            if not isinstance(sampled_indices, list):
                sampled_indices = [sampled_indices]
            if not isinstance(sampled_tokens, list):
                sampled_tokens = [sampled_tokens]
                
            latent = model.encoder(inputs - model.decoder_bias) + model.encoder_bias # (batch_size, hidden_dim)

            # Collect activations for each feature
            for feature_idx in feature_indices:
                feature_activations = latent[:, feature_idx].cpu().numpy()
                for j, (activation, text, indices, token) in enumerate(zip(feature_activations, prompt_texts, sampled_indices, sampled_tokens)):
                    index = indices[j].item()  # Get the specific index for this iteration
                    top_activations_data[feature_idx].append((activation, text, index, token))

            # Sort and truncate top_activations_data
            if (i * num_features) % (10 * num_features) == 0:
                # Sort activations for each feature
                for feature_idx in top_activations_data:
                    top_activations_data[feature_idx].sort(key=lambda x: x[0], reverse=True)
                    top_activations_data[feature_idx] = top_activations_data[feature_idx][:top_activations]

    # Sort activations for each feature one last time
    for feature_idx in top_activations_data:
        top_activations_data[feature_idx].sort(key=lambda x: x[0], reverse=True)
        top_activations_data[feature_idx] = top_activations_data[feature_idx][:top_activations]

    # Create DataFrame (directly from sorted data)
    results_data = []
    for feature_idx, activations in top_activations_data.items():
        results_data.extend([
            {"Feature": feature_idx, "Activation Rank": j + 1, "Activation": val, "Full Text": text, "Token Index": index, "Token Char": token}
            for j, (val, text, index, token) in enumerate(activations)
        ])
    df = pd.DataFrame(results_data)

    # Save to CSV
    filename = f"sae_interpretations_iter{iteration}.csv" if iteration is not None else "sae_interpretations.csv"
    df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Clean up CUDA memory
    torch.cuda.empty_cache()


def get_feature_activations(model, prompts, feature_index, tokenizer):
    if isinstance(prompts, str):
        prompts = [prompts]

    device = next(model.parameters()).device
    activations_list = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]

            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].cpu()

            num_pad_tokens = input_ids.shape[1] - torch.sum(attention_mask).item()
            valid_tokens = hidden_states.size(1) - num_pad_tokens

            latent = model.encoder(hidden_states - model.decoder_bias) + model.encoder_bias
            feature_activations = latent[0, num_pad_tokens:, feature_index].cpu().numpy().tolist()

            activations_list.append(feature_activations)

    return activations_list