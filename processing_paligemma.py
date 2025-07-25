from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Constructs a text prompt that simulates a multi-modal (image + text) input.

    This function prepends a series of special <image> tokens to the actual text prompt,
    followed by a <bos> (beginning-of-sequence) token and a newline `\n` at the end.
    This format is required by some models (like PaLI-Gemma) during inference.

    Args:
        prefix_prompt (str): The actual human-readable prompt (e.g., "Describe the image").
        bos_token (str): Special token representing the start of a text sequence (e.g., "<bos>").
        image_seq_len (int): Number of <image> tokens to prepend (represents encoded visual input).
        image_token (str): The <image> token string to use (e.g., "<image>").

    Returns:
        str: Final input string of the form:
             "<image><image>...<image><bos>prefix_prompt\n"
             Example: "<image><image><image><bos>Describe the image\n"
    """

    # Add `image_seq_len` number of <image> tokens (simulates vision input)
    image_tokens = image_token * image_seq_len

    # Add <bos> token to mark the beginning of text
    # Add the actual text prompt
    # Append a newline character (`\n`) because the model was trained with it as part of the prompt
    final_prompt = f"{image_tokens}{bos_token}{prefix_prompt}\n"

    return final_prompt


def rescale(
    image: np.ndarray, 
    scale: float, 
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescales the pixel values of the image by a given scale factor.

    Args:
        image (np.ndarray): Input image array of shape [C, H, W] or [H, W, C].
        scale (float): Factor to multiply pixel values by (e.g., 1/255.0 to normalize to [0, 1]).
        dtype (np.dtype): Desired output data type (default is float32).

    Returns:
        np.ndarray: Rescaled image with the same shape as input and dtype = `dtype`.
    """
    # Multiply each pixel by the scale factor — typically used to bring values into [0, 1] range
    rescaled_image = image * scale
    
    # Ensure consistent numeric type (e.g., float32 for neural network input)
    rescaled_image = rescaled_image.astype(dtype)
    
    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """
    Resizes the given PIL image to the specified dimensions.

    Args:
        image (PIL.Image.Image): Input image in PIL format.
        size (Tuple[int, int]): Target size as (height, width).
        resample (Image.Resampling): Resampling algorithm (e.g., BICUBIC, BILINEAR).
        reducing_gap (Optional[int]): Optional parameter to optimize resizing.

    Returns:
        PIL.Image.Image: Resized image.
    """
    # Unpack the target size — note that PIL expects size as (width, height)
    height, width = size

    # Resize the image with the given resampling method (e.g., BICUBIC for smooth results)
    resized_image = image.resize(
        (width, height),  # PIL uses (width, height) order
        resample=resample,
        reducing_gap=reducing_gap  # Improves quality and speed when downsampling
    )
    
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalizes the image by subtracting mean and dividing by standard deviation.

    Args:
        image (np.ndarray): Input image array, typically of shape [C, H, W] or [H, W, C].
        mean (float or Iterable[float]): Mean values (either global or per channel).
        std (float or Iterable[float]): Standard deviation values (global or per channel).

    Returns:
        np.ndarray: Normalized image with same shape and dtype as input.
    """
    # Convert mean and std to numpy arrays and ensure they match the image's data type
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    # Normalize each pixel/channel: (x - mean) / std
    # If mean/std are per-channel, broadcasting will be applied across spatial dimensions
    image = (image - mean) / std

    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Preprocesses a list of PIL images by resizing, rescaling, normalizing, and reordering channels.

    Args:
        images (List[Image.Image]): List of input PIL images.
        size (Dict[str, int]): Dictionary with keys 'height' and 'width' specifying the target image size.
        resample (Image.Resampling): Resampling method used during resizing (e.g., Image.BILINEAR).
        rescale_factor (float): Value by which pixel values will be divided to bring them into [0, 1] range.
        image_mean (float or List[float], optional): Mean value(s) used to normalize the image (channel-wise or global).
        image_std (float or List[float], optional): Standard deviation(s) used to normalize the image (channel-wise or global).

    Returns:
        List[np.ndarray]: List of preprocessed images as NumPy arrays in CHW (channel, height, width) format.
    """
    
    # Extract target height and width from the size dictionary
    height, width = size["height"], size["width"]

    # Step 1: Resize each image to the target dimensions using the specified resampling method
    images = [
        resize(image=image, size=(height, width), resample=resample)
        for image in images
    ]

    # Step 2: Convert each resized image from PIL to a NumPy array
    images = [np.array(image) for image in images]

    # Step 3: Rescale pixel values to a [0, 1] range if a rescale factor is provided
    # This is typically 1/255 if the image was in [0, 255]
    images = [rescale(image, scale=rescale_factor) for image in images]

    # Step 4: Normalize each image using the specified mean and std values
    # This helps center the data and ensures better convergence during training
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # Step 5: Convert image shape from HWC (height, width, channels) to CHW (channels, height, width)
    # This is the format expected by most deep learning models (e.g., PyTorch, some transformers)
    images = [image.transpose(2, 0, 1) for image in images]

    return images



class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"  

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        # Number of tokens that will represent the image in the input sequence
        # This is determined by the image encoder (e.g., ViT) and depends on the patch size and image resolution.
        # For example, for a 224x224 image with a 16x16 patch size, this would be 14*14 = 196 tokens.
        self.image_seq_length = num_image_tokens

        # Store the input image size (e.g., (224, 224)); useful for resizing or positional encoding
        self.image_size = image_size

        # --- Tokenizer Augmentation ---

        # Add a special image token to represent the presence or position of an image in the sequence
        # This token is typically prepended or inserted into the input to indicate where image features are aligned
        # Tokenizer details reference: 
        # https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # Add special tokens for spatial object localization
        # <loc0000> to <loc1023>: used to represent bounding box anchors for object detection
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]

        # Add special tokens for image segmentation
        # <seg000> to <seg127>: used to identify different segments or masks in the image
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)

        # Get and store the token ID corresponding to the special image token
        # This is useful for checking or injecting the token into input sequences later
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Disable automatic addition of BOS (beginning-of-sequence) and EOS (end-of-sequence) tokens by the tokenizer
        # We'll handle these manually when constructing model inputs
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        # Store the modified tokenizer for downstream usage
        self.tokenizer = tokenizer


    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        # Ensure that exactly one image is provided per one text prompt.
        # If not, raise an AssertionError.
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Step 1: Process the input image(s)
        # - Resize to (image_size, image_size)
        # - Normalize using ImageNet mean and std
        # - Resample using bicubic interpolation
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Step 2: Convert list of processed images (each a NumPy array of shape [C, H, W]) 
        # into a single stacked array of shape [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)  # Shape: [1, 3, H, W]

        # Step 3: Convert the NumPy array into a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)  # Shape: [1, 3, H, W], dtype: torch.float32

        # Step 4: Modify the text prompt by prepending image tokens to simulate multimodal input
        # This will look like: "<bos><image><image>...<image><actual_text>"
        # Number of <image> tokens = self.image_seq_length
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Step 5: Tokenize the modified text prompts
        # Returns a dictionary with:
        # - input_ids: Tensor of shape [B, Seq_Len]
        # - attention_mask: Tensor of shape [B, Seq_Len]
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",     # Return PyTorch tensors
            padding=padding,         # Pad sequences to the longest one (or user-defined strategy)
            truncation=truncation,   # Truncate sequences that are too long
        )

        # Step 6: Combine image tensor with tokenized text
        return_data = {
            "pixel_values": pixel_values,  # Shape: [1, 3, H, W]
            **inputs                       # input_ids and attention_mask
        }

        return return_data
