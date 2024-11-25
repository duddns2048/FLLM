import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification

# Load the ViT model and feature extractor with higher input resolution
model_name = "google/vit-base-patch16-384"
model = ViTModel.from_pretrained(model_name, output_attentions=True)
# model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Load and preprocess the image
image = Image.open("./examples/dog2.jpg")  # replace with your image path
inputs = feature_extractor(images=image, return_tensors="pt")
image_tensor = inputs['pixel_values']
# Forward pass through the model to extract feature activations
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, num_tokens, hidden_size)
    pooler_output = outputs.pooler_output
    attentions = outputs.attentions

# Extract the feature activation for all tokens except [CLS]
# Reshape to match the image grid (24x24 for ViT base model with patch size 16 and input 384x384)

# Average of feature activation
feature_activations = last_hidden_state[0, 1:, :].numpy()  # Exclude [CLS] token
avg_activations = np.mean(feature_activations, axis=-1)
activation_map = avg_activations.reshape(24, 24)

# Magnitude of feature activation
visual_tokens = last_hidden_state[0, 1:, :]  # Exclude [CLS] token
avg_token_feature = visual_tokens.mean(dim=0, keepdim=True)
activation = (visual_tokens - avg_token_feature).norm(dim=-1)

mag_min, mag_max = activation.min(), activation.max()
mag_activation = (activation - mag_min) / (mag_max - mag_min)
activation_map = mag_activation.reshape(24, 24).numpy()

# frequency of feature activation
freq_token = torch.fft.fft(visual_tokens).angle()
avg_freq_token = freq_token.mean(dim=0, keepdim=True)
frequency_activation = (freq_token - avg_freq_token).norm(dim=-1)

freq_min, freq_max = frequency_activation.min(), frequency_activation.max()
freq_activation = (frequency_activation - freq_min) / (freq_max - freq_min)
freq_activation_map = freq_activation.reshape(24, 24).numpy()



# Get original and processed image shapes
original_shape = image.size  # (width, height)
processed_shape = image_tensor.shape[-2:]  # (height, width)

# Visualize attention scores between [CLS] token and visual tokens
# Extract the attention weights for the [CLS] token from the last layer
cls_attention_map = attentions[-1][0, :, 0, 1:].numpy()  # Shape: (num_heads, num_tokens - 1)

# Average over all heads to simplify visualization
avg_cls_attention_map = np.mean(cls_attention_map, axis=0)

# Reshape to 24x24 to match the image grid
cls_attention_map_reshaped = avg_cls_attention_map.reshape(24, 24)

# Create a single figure to save all visualizations
fig, axes = plt.subplots(3, 2, figsize=(20, 30))
# fig, axes = plt.subplots(3, 2, figsize=(20, 30), gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1, 1]})
plt.rcParams.update({'font.size': 18})

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title(f"Original Image (Resolution: {original_shape[0]}x{original_shape[1]})")
axes[0, 0].axis('off')

# Feature extractor processed image
axes[0, 1].imshow(image_tensor[0].permute(1, 2, 0))
axes[0, 1].set_title(f"Feature Extractor Processed Image (Resolution: {processed_shape[1]}x{processed_shape[0]})")
axes[0, 1].axis('off')

# Feature activation map
cax1 = axes[1, 0].imshow(activation_map, cmap='viridis')
axes[1, 0].set_title("ViT Feature Activation Map \n (Magnitude Normalized) with Input Size 384")
axes[1, 0].axis('off')
fig.colorbar(cax1, ax=axes[1, 0], fraction=0.046, pad=0.04)


# Feature activation map (frequency)
cax2 = axes[1, 1].imshow(freq_activation_map, cmap='viridis')
axes[1, 1].set_title("ViT Feature Activation Map \n (Frequency Normalized) with Input Size 384")
axes[1, 1].axis('off')
fig.colorbar(cax2, ax=axes[1, 1], fraction=0.046, pad=0.04)

# [CLS] token attention map
cax3 = axes[2, 0].imshow(cls_attention_map_reshaped, cmap='viridis')
axes[2, 0].set_title("Attention Scores between [CLS] Token and Visual Tokens \n (Last Layer, Averaged over Heads)")
axes[2, 0].axis('off')
fig.colorbar(cax3, ax=axes[2, 0], fraction=0.046, pad=0.04)

# Leave the last subplot empty
axes[2, 1].axis('off')

# Save the figure
# plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.show()
plt.savefig("visualizations_combined_dog2.png")