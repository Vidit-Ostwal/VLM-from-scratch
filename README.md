# VLM-from-scratch

This repository is part of my personal learning journey to understand **Vision-Language Models (VLMs)** — specifically, how models interpret an image and generate meaningful captions from it.

My curiosity stemmed from a simple question:

> _"How does a model actually understand an image and then generate a relevant caption for it?"_

This project is built alongside [Umar Jabil’s excellent video](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=11763s), which I highly recommend watching.  
**All credit goes to him for the concepts and guidance.**

---

## 📁 File Overview

### `modelling_siglip.py`

This file implements the Vision Transformer (ViT) portion of the SigLIP architecture. It includes:

1. **`SigLIPVisionConfig`**  
   Configuration class for setting model parameters.

2. **`SigLIPVisionEmbeddings`**  
   Responsible for converting image patches into embedding vectors.

3. **`SigLIPMLP`**  
   Standard feed-forward network used within transformer blocks.

4. **`SigLIPAttention`**  
   Multi-head self-attention mechanism for capturing spatial relationships.

5. **`SigLIPEncoderLayer`**  
   A single transformer encoder block: Attention + MLP + LayerNorm.

6. **`SigLIPEncoder`**  
   Stacked encoder layers forming the transformer backbone.

7. **`SigLIPVisionTransformer`**  
   Combines embeddings and encoder to form the ViT.

8. **`SigLIPVisionModel`**  
   High-level model class wrapping the Vision Transformer.

---

### `processing_paligemma.py`

This file provides a lightweight preprocessing module for preparing image and text inputs for PaLI-Gemma-style vision-language models. It includes:

1. **`add_image_tokens_to_prompt`**  
   Prepends `<image>` tokens and a `<bos>` token to the prompt, and appends a newline for compatibility with training format.

2. **`rescale`**  
   Scales pixel values (e.g., by `1/255.0`) and converts the array to a float dtype.

3. **`resize`**  
   Resizes a `PIL.Image` to a target `(height, width)` using the specified resampling method.

4. **`normalize`**  
   Normalizes an image by subtracting the mean and dividing by the standard deviation.

5. **`process_images`**  
   Applies resize → CHW conversion → rescaling → normalization to a list of images.

6. **`PaligemmaProcessor`**  
   Main callable that processes images and text into `pixel_values`, `input_ids`, and `attention_mask` for model input.



## 🚧 Notes

- This repository is **educational** in nature and may not be optimized for performance.
- You are encouraged to tweak, explore, and build on top of this to better understand the internals of vision-language models.
- Check out the linked video to follow along with the architecture and coding.

---

Happy building! 🛠️  
Feel free to fork, contribute, or drop suggestions.
