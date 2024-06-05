import torch
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm


def evaluate_sae(model, dataloader, num_features=5, top_activations=10, iteration=None):
    model.eval()
    device = next(model.parameters()).device

    top_activations_data = {}
    failed_batches = []

    progress_bar = tqdm(dataloader, desc="Evaluating SAE", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            try:
                inputs = batch['activations'].to(device)
                prompt_texts = batch["prompt_texts"]  
                sampled_indices = batch["sampled_indices"]  

                # Ensure these are lists for consistent indexing
                if not isinstance(prompt_texts, list):
                    prompt_texts = [prompt_texts]
                if not isinstance(sampled_indices, list):
                    sampled_indices = [sampled_indices]
                    
                latent = model.encoder(inputs - model.decoder_bias) + model.encoder_bias

                for feature_idx in range(latent.size(1)):
                    feature_activations = latent[:, feature_idx].cpu().numpy()
                    top_indices = np.argsort(feature_activations)[::-1][:top_activations]
                    top_activations_values = feature_activations[top_indices]
                    
                    # Ensure indices are valid for prompt_texts and sampled_indices
                    valid_indices = [idx for idx in top_indices if idx < len(prompt_texts) and idx < len(sampled_indices)]

                    # Handle the case where no valid indices are found
                    if not valid_indices:
                        print(f"Warning: No valid indices found for batch {i}, feature {feature_idx}. Skipping.")
                        continue  # Move to the next feature

                    top_activations_data.setdefault(feature_idx, []).extend(
                        [(val, prompt_texts[idx], sampled_indices[idx][0]) for idx, val in zip(valid_indices, top_activations_values[valid_indices])]
                    )

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Warning: Batch {i} caused CUDA OOM. Skipping and clearing cache.")
                    failed_batches.append(i)
                    del inputs, prompt_texts, sampled_indices, latent 
                    torch.cuda.empty_cache() 
                else:
                    raise e  # Re-raise other exceptions
            finally:  
                # Ensure progress bar updates and is closed.
                if (i + 1) % 500 == 0:
                    progress_bar.update(500)

    progress_bar.close()

    # Create DataFrame (directly from sorted data)
    results_data = []
    for feature_idx, activations in top_activations_data.items():
        results_data.extend([
            {"Feature": feature_idx, "Rank": j + 1, "Activation": val, "Text": text, "Token": token}
            for j, (val, text, token) in enumerate(activations)
        ])
    df = pd.DataFrame(results_data)

    # Save to CSV
    filename = f"sae_interpretations_iter{iteration}.csv" if iteration is not None else "sae_interpretations.csv"
    df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)

    if failed_batches:
        print(f"Warning: Failed to process batches: {failed_batches} due to CUDA OOM.")



# import torch
# import numpy as np
# import csv
# import pandas as pd
# from tqdm import tqdm

# def evaluate_sae(model, dataloader, num_features=5, top_activations=10, iteration=None):
#     model.eval()
#     device = next(model.parameters()).device

#     # Initialize storage
#     top_activations_data = {}  

#     progress_bar = tqdm(dataloader, desc="Evaluating SAE", leave=False, dynamic_ncols=True)
    
#     with torch.no_grad():
#         for i, batch in enumerate(progress_bar):
#             inputs = batch['activations'].to(device)
#             prompt_texts = batch["prompt_texts"]
#             sampled_indices = batch["sampled_indices"].cpu().numpy()  
#             latent = model.encoder(inputs - model.decoder_bias) + model.encoder_bias  

#             # Iterate over all features
#             # for feature_idx in range(latent.size(1)):
#             #     feature_activations = latent[:, feature_idx].cpu().numpy()
#             #     top_indices = np.argsort(feature_activations)[::-1][:top_activations] 
#             #     top_activations_values = feature_activations[top_indices]

#             #     # Inside the loop, after collecting top_activations_values
#             #     for feature_idx, activations in top_activations_data.items():
#             #         top_activations_data[feature_idx] = [
#             #             (val, text, token, tokenizer.decode(token).replace(" ", "_"))  # Decode and replace spaces
#             #             for val, text, token in activations
#             #         ]

#             for feature_idx in range(latent.size(1)):
#                 feature_activations = latent[:, feature_idx].cpu().numpy()
#                 top_indices = np.argsort(feature_activations)[::-1][:top_activations]
#                 top_activations_values = feature_activations[top_indices]
    
#                 # Directly store top activations and associated data (outside the inner loop)
#                 top_activations_data.setdefault(feature_idx, []).extend(
#                     [(val, prompt_texts[idx], sampled_indices[idx][0]) for idx, val in zip(top_indices, top_activations_values)]
#                 )

#             # Update progress bar every 500 iterations
#             if (i + 1) % 500 == 0:
#                 progress_bar.update(500)
            
#     progress_bar.close()  # Make sure to close the progress bar at the end


#     # Create DataFrame (directly from sorted data)
#     results_data = []
#     for feature_idx, activations in top_activations_data.items():
#         results_data.extend([
#             {"Feature": feature_idx, "Rank": j + 1, "Activation": val, "Text": text, "Token": token, "Character": char}
#             for j, (val, text, token, char) in enumerate(activations)
#         ])

#     df = pd.DataFrame(results_data)

#     # Save to CSV
#     filename = f"sae_interpretations_iter{iteration}.csv" if iteration is not None else "sae_interpretations.csv"
#     df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)