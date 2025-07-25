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

## 🚧 Notes

- This repository is **educational** in nature and may not be optimized for performance.
- You are encouraged to tweak, explore, and build on top of this to better understand the internals of vision-language models.
- Check out the linked video to follow along with the architecture and coding.

---

Happy building! 🛠️  
Feel free to fork, contribute, or drop suggestions.
