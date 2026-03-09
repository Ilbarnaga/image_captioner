"""
Cost Evaluator Module.

This module provides the `CostEvaluator` class, which calculates the projected
API costs for processing a dataset of images through various Vision-Language Models.
It uses provider-specific tokenization formulas to ensure accurate estimates.
"""

import math
from pathlib import Path


class CostEvaluator:
    """
    Calculates costs using strictly active model data and pricing rates.
    
    This class scans a target dataset directory for unprocessed images, calculates
    the expected number of input and output tokens based on the active AI model's 
    specific vision tokenization rules, and estimates the total USD cost.
    """

    def __init__(
            self,
            model_name      : str,
            pricing         : dict,
            max_tokens      : int,
            dataset_path    : Path,
            img_res         : int,
            text_overhead   : int = 250
        ):
        """
        Initializes the CostEvaluator with model details and pricing metrics.

        Args:
            model_name (str): The identifier of the active AI model (e.g., 'gpt-4o', 'claude-3-5-sonnet').
            pricing (dict): A dictionary containing 'input' and 'output' cost per 1 million tokens in USD.
            max_tokens (int): The maximum number of tokens expected in the output response.
            dataset_path (Path): The directory path containing the dataset images.
            img_res (int): The maximum resolution constraint applied to the images during preprocessing.
            text_overhead (int, optional): Estimated tokens for system/user instructions. Defaults to 250.
        """
        self.model          : str  = model_name.lower()
        self.pricing        : dict = pricing  # Format: {'input': float, 'output': float}
        self.max_tokens     : int  = max_tokens
        self.dataset_path   : Path = dataset_path
        self.img_res        : int  = img_res
        self.text_overhead  : int  = text_overhead # Estimated tokens for system/user instructions

    def _estimate_vision_tokens(self, width: int, height: int) -> int:
        """
        Calculates vision tokens based on provider-specific tiling formulas.

        Args:
            width (int): The width of the scaled image.
            height (int): The height of the scaled image.

        Returns:
            int: The estimated number of input tokens consumed by the image.
        """
        # OpenAI & Grok High-Detail: 170 per 512px tile + 85 base
        if any(x in self.model for x in ['gpt', 'grok']):
            scale = 768 / min(width, height)
            w, h = width * scale, height * scale
            tiles = math.ceil(w / 512) * math.ceil(h / 512)
            return (tiles * 170) + 85
            
        # Claude Formula: Total Pixels / 750
        elif 'claude' in self.model:
            return math.ceil((width * height) / 750)
            
        return 1200 # Standard baseline fallback

    def _get_unprocessed_count(self) -> int:
        """
        Counts images that do not yet have a .txt caption file.

        Returns:
            int: The count of unprocessed image files in the dataset directory.
        """
        if not self.dataset_path.exists(): 
            return 0
            
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
        
        # Iterate through the directory and filter for images lacking a matching text file
        return len([
            f for f in self.dataset_path.iterdir() 
            if f.suffix.lower() in valid_exts and not f.with_suffix('.txt').exists()
        ])

    def calculate(self) -> dict:
        """
        Outputs a dictionary of required tokens and USD cost.

        Returns:
            dict: A structured dictionary containing image counts, token breakdowns,
                  and total estimated costs in USD. Returns an error dict if no new images are found.
        """
        image_count = self._get_unprocessed_count()
        
        if image_count <= 0:
            return {"error": "No new images found for processing."}

        # Calculate Tokens
        # Assuming square resolution based on the app config constraint (img_res x img_res)
        tokens_in_per_img = self._estimate_vision_tokens(self.img_res, self.img_res) + self.text_overhead
        total_tokens_in = image_count * tokens_in_per_img
        total_tokens_out = image_count * self.max_tokens

        # Calculate USD (Pricing in config is per 1 Million tokens)
        input_usd = (total_tokens_in / 1_000_000) * self.pricing['input']
        output_usd = (total_tokens_out / 1_000_000) * self.pricing['output']

        return {
            "images": image_count,
            "tokens": {
                "input": total_tokens_in,
                "output": total_tokens_out,
                "total": total_tokens_in + total_tokens_out
            },
            "usd": {
                "input": input_usd,
                "output": output_usd,
                "total": input_usd + output_usd
            }
        }