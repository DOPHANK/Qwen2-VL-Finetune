import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    """
    Replace LLaVA-style <image>/<video> tokens with OpenAI-style vision tokens.

    Args:
        input_string (str): Input text containing LLaVA tokens.
        is_video (bool): If True, replace video tokens, otherwise replace image tokens.

    Returns:
        str: Text with tokens replaced.
    """

    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    """
    Convert LLaVA conversation format to OpenAI conversation format.

    Args:
        conversations (list[dict]): List of conversation dicts with keys {"from","value"}.
        is_video (bool): Whether the conversations involve videos.

    Returns:
        list[dict]: Converted list with {"role","content"} format.
    """

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    """
    Truncate input/label tensors to max_length and append EOS if provided.

    Args:
        input_ids (torch.Tensor): Input token IDs.
        labels (torch.Tensor): Label token IDs.
        max_length (int): Maximum allowed length.
        eos_token_id (int): EOS token ID to append.

    Returns:
        (torch.Tensor, torch.Tensor): Truncated input_ids and labels.
    """

    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of variable-length tensors to the same length.

    Args:
        sequences (list[torch.Tensor]): List of tensors [seq_len, *].
        padding_side (str): 'right' or 'left'.
        padding_value (int): Value to pad with.

    Returns:
        torch.Tensor: Batch tensor of shape (batch, max_len, ...).
    """

    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    """
    Create vision input dict for an image and process it with Qwen utilities.

    Args:
        image_path (str): Path to the image.
        min_pixel (int): Minimum pixels allowed.
        max_pixel (int): Maximum pixels allowed.
        width (int): Optional resized width.
        height (int): Optional resized height.

    Returns:
        torch.Tensor: Processed image tensor.
    """

    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    """
    Create vision input dict for a video and process it with Qwen utilities.

    Args:
        video_path (str): Path to the video.
        min_pixels (int): Minimum pixel count.
        max_pixels (int): Maximum pixel count.
        width (int): Resized width.
        height (int): Resized height.
        fps (int): Frames per second.

    Returns:
        (torch.Tensor, dict): Video tensor and kwargs for video processing.
    """

    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs
