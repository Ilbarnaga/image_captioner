"""
Core Captioner Orchestrator.

This module provides the `Captioner` class, which manages the asynchronous pipeline
for generating, validating, and saving image captions using Vision-Language Models (VLMs).
It handles configuration loading, client initialization, and concurrent API request execution.
"""

import base64
import asyncio
import sys
import yaml
from pathlib import Path
from io import BytesIO
from PIL import Image

# --- AI Client Imports ---
from src.ai_clients.base_client import AsyncVisionClient
from src.ai_clients.openai_client import OpenAIVisionClient
from src.ai_clients.claude_client import ClaudeVisionClient

# --- Utility Imports ---
from src.utils.cost_evaluator import CostEvaluator
from src.utils.quality_checker import QualityChecker, CaptionValidationError
from src.utils.colors import CYAN, VIOLET, ORANGE, YELLOW, GREEN, RED, BOLD, NC


class Captioner:
    """
    Manages the async pipeline, quality control, and auto-retries for dataset captioning.

    This class orchestrates the entire workflow: loading settings from a YAML file,
    initializing the appropriate AI Vision client, applying strict text validations 
    via the QualityChecker, and processing images concurrently to maximize throughput.
    """
    
    def __init__(self):
        """
        Initializes the Captioner instance, establishes base directory paths,
        loads the YAML configuration, and sets up the required operational tools.
        """
        print(f"\n{BOLD}{CYAN}🚀 Initializing AI Captioner")
        
        # --- Path Configurations ---
        self.root            : Path = Path(__file__).resolve().parent.parent.parent
        self.config_path     : Path = self.root / "config" / "config.yaml"
        self.prompt_path     : Path = self.root / "src" / "prompting" / "captioning_prompt.txt"
        self.dataset_dir     : Path = self.root / "dataset"
        
        # --- State Variables ---
        self.config          : dict = {}
        self.prompt          : str  = ""

        # Load configurations and initialize external tools
        self._load_config()
        self._initialize_tools()

        # --- Concurrency Control ---
        # Limits the number of simultaneous API calls to avoid rate limiting
        self.semaphore       : asyncio.Semaphore = asyncio.Semaphore(self.config['app']['max_concurrent_requests'])

    def _load_config(self) -> dict:
        """
        Reads and parses the primary configuration YAML file.

        Raises:
            SystemExit: If the configuration file cannot be found at `self.config_path`.
        """
        if not self.config_path.exists():
            print(f"{RED}🚨 Config file not found at {BOLD}{self.config_path}!{NC}\n")
            sys.exit(1)
            
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            print(f"{GREEN}✅ Loaded configuration from {BOLD}{self.config_path.name}!{NC}")

    def _save_config(self) -> None:
        """
        Persists the current in-memory configuration dictionary back to the YAML file,
        and re-initializes tools to reflect the updated settings.
        """       
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"{GREEN}✅ Configuration saved successfully to {BOLD}{self.config_path.name}!{NC}")
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """
        Instantiates or re-instantiates the core operational tools based on the 
        current configuration, including the AI client, quality checker, and cost evaluator.
        """
        self.ai_client       : AsyncVisionClient = self._create_ai_client()
        self.checker         : QualityChecker    = self._create_quality_checker()
        self.cost_evaluator  : CostEvaluator     = self._create_cost_evaluator()
        self.prompt          : str               = self._load_and_prepare_prompt()
        print()

    def _create_ai_client(self) -> AsyncVisionClient:
        """
        Dynamically instantiates the correct Vision AI client based on the active
        provider listed in the configuration file.

        Returns:
            AsyncVisionClient: An initialized instance of the selected AI client.

        Raises:
            ValueError: If an unsupported API provider is specified.
        """
        # Extract active routing settings
        active = self.config['api'].get('active').get('provider').lower()
        provider_settings = self.config['api'].get('providers').get(active)
        api_key = provider_settings.get('api_key')
        
        # Extract model parameters
        model = self.config['api']['active']['model']
        max_tokens = self.config['api'].get('max_tokens')
        temperature = self.config['api'].get('temperature')

        if not api_key:
            print(f"{RED}❌ API key is missing for the active provider: {BOLD}{active}{NC}{RED}!{NC}")
            return
        else:
            print(f"{VIOLET}🌐 Initializing AI client for {ORANGE}{BOLD}{model}{NC}{VIOLET} "
                  f"(max tokens: {ORANGE}{max_tokens}{VIOLET}, temperature: {ORANGE}{temperature}{VIOLET}){NC}")

        # Route to OpenAI or Grok (Since they share the OpenAI SDK structure)
        if active in ['openai', 'grok', 'xai']:
            base_url = provider_settings.get('base_url')
            return OpenAIVisionClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
        # Route to Anthropic/Claude
        elif active in ['claude', 'anthropic']:
            return ClaudeVisionClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
        else:
            raise ValueError(f"Unsupported active API provider: '{active}'")

    def _create_quality_checker(self) -> QualityChecker:
        """
        Initializes the QualityChecker, which validates AI outputs against LoRA rules.

        Returns:
            QualityChecker: Configured to enforce word count and trigger word usage.
        """
        trigger_word = self.config['lora']['trigger_word']
        min_words = self.config['lora']['min_words']
        max_words = self.config['lora']['max_words']
        
        print(f"{VIOLET}🎯 Initializing quality checker with trigger word {ORANGE}{BOLD}{trigger_word}{NC}{VIOLET} "
              f"(min words: {ORANGE}{min_words}{VIOLET}, max_words: {ORANGE}{max_words}{VIOLET}){NC}")
        
        return QualityChecker(
            trigger_word=trigger_word,
            min_words=min_words,
            max_words=max_words
        )

    def _create_cost_evaluator(self) -> CostEvaluator:
        """
        Initializes the CostEvaluator to calculate projected API usage expenses.

        Returns:
            CostEvaluator: Configured with current model pricing and dataset parameters.
        """
        api_cfg = self.config['api']
        provider = api_cfg['active']['provider']
        model = api_cfg['active']['model']
        
        # Get specific pricing for the active model
        pricing = api_cfg['providers'][provider]['pricing'].get(model)
        max_tokens = api_cfg.get('max_tokens')
        
        dataset_path = Path(self.config['app']['dataset_dir'])
        img_res = self.config['app'].get('max_image_size')

        print(f"{VIOLET}💵 Initializing cost evaluator for {ORANGE}{BOLD}{model}{NC}")
        
        return CostEvaluator(
            model_name=model,
            pricing=pricing,
            max_tokens=max_tokens,
            dataset_path=dataset_path,
            img_res=img_res
        )

    def _load_and_prepare_prompt(self) -> str:
        """
        Loads the foundational text prompt from disk and injects specific configurations
        like the trigger word and target word counts.

        Returns:
            str: The fully formatted prompt ready to be sent to the AI.
            
        Raises:
            SystemExit: If the text prompt file cannot be located.
        """
        if not self.prompt_path.exists():
            print(f"{RED}🚨 Prompt file not found at {BOLD}{self.prompt_path}!{NC}")
            sys.exit(1)

        with self.prompt_path.open("r", encoding="utf-8") as file:
            raw_prompt = file.read()
            
        trigger_word = self.config['lora']['trigger_word']
        min_words = self.config['lora']['min_words']
        max_words = self.config['lora']['max_words']
        
        print(f"📝 {VIOLET}Loaded prompt from {ORANGE}{BOLD}{self.prompt_path.name}{NC}")
        
        return raw_prompt.replace("[TRIGGER_WORD]", trigger_word)\
                         .replace("[MIN_WORDS]", str(min_words))\
                         .replace("[MAX_WORDS]", str(max_words))


    # ===== PIPELINE EXECUTION =====
    def _process_and_encode(self, file_path: str) -> str:
        """
        Handles CPU-bound image operations safely.
        Opens, resizes (maintaining aspect ratio), and converts the image to base64.
        Designed to be run in a thread executor to prevent blocking the async loop.
        """
        with Image.open(file_path) as img:
            # Convert to RGB to avoid issues with RGBA/PNGs
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new dimensions while maintaining aspect ratio
            img.thumbnail(
                (
                    self.config.get('app').get('max_image_size'),
                    self.config.get('app').get('max_image_size')
                ),
                Image.Resampling.LANCZOS
            )
            
            # Save to memory buffer
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            # Encode to base64
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded_string
        
    async def _process_single_image(self, file_path: Path) -> None:
        """
        The core pipeline logic for a single image. Handles preprocessing, API requests,
        quality validation, auto-correction loops, and file output.

        Args:
            file_path (Path): The absolute or relative path to the image file.
        """
        txt_path = file_path.with_suffix('.txt')
        
        # Native checkpointing: Skip this image if a caption file already exists
        if txt_path.exists():
            return
            
        async with self.semaphore:
            print(f"🔄 {VIOLET}Processing: {BOLD}{ORANGE}{file_path.name}{NC}")
            try:
                # 1. Threaded Image Preprocessing (resizing and base64 encoding)
                image_b64 = await asyncio.to_thread(
                    self._process_and_encode, str(file_path)
                )

                # 2. Self-Correcting Retry Loop
                attempts = 0
                final_caption = ""
                
                # Create a localized copy of the prompt to safely modify it on failures
                current_prompt = self.prompt 
                
                while attempts < self.config['app']['max_retries']:
                    attempts += 1
                    
                    # Call the AI provider with the localized prompt
                    caption = await self.ai_client.generate_caption(image_b64, current_prompt) 
                    
                    try:
                        # 3. Quality Check Validation
                        self.checker.validate(caption)
                        final_caption = caption
                        break  # Passed QC, break the retry loop
                        
                    except CaptionValidationError as e:
                        # Log the QC failure
                        print(f"{YELLOW}⚠️  QC Failed on {BOLD}{file_path.name}{NC}{YELLOW} "
                              f"(Attempt {attempts}/{self.config['app']['max_retries']}): {e}")
                        
                        if attempts == self.config['app']['max_retries']:
                            print(f"{RED}❌ Max retries reached for {file_path.name}. Skipping.")
                            return
                        
                        # Append the failure reason back to the prompt for the next loop to force correction
                        current_prompt += f"\n\nYOUR PREVIOUS ATTEMPT FAILED THE QUALITY CHECK: {str(e)}\nFix this immediately."
                
                # 4. Save Final Output
                if final_caption:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_caption)
                    print(f"{GREEN}✅ Success: caption for file {BOLD}{file_path.name}{NC}{GREEN} generated!{NC}")

            except Exception as e:
                print(f"{RED}❌ Error processing {BOLD}{file_path.name}{NC}{RED}: {str(e)}")

    async def run(self) -> None:
        """
        Scans the target directory for valid images and initiates the asynchronous 
        captioning pipeline for all discovered files.
        """
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        
        # Identify all target images in the dataset directory
        files = [f for f in self.dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

        print(f"\n{CYAN} ===== Found {BOLD}{len(files)}{NC}{CYAN} images. Starting captioning ====={NC}")
        
        # Gather and execute all processing tasks concurrently
        await asyncio.gather(*(self._process_single_image(f) for f in files))
        
        print (f"====={GREEN}{BOLD} 🎉 All captioning complete! ===== {NC}\n")