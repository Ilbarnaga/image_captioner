"""
Command Line Interface (CLI) Module.

This module provides the interactive terminal interface for the AI Image Captioner.
It utilizes InquirerPy to render rich, interactive menus and forms, allowing users
to seamlessly configure settings, evaluate costs, and execute the captioning pipeline.
"""

import os
import platform
import subprocess
import sys
from typing import Callable, Coroutine

from InquirerPy import inquirer
from InquirerPy.utils import get_style
from pyfiglet import Figlet

# --- Local Imports ---
from src.core.captioner import Captioner
from src.utils.colors import CYAN, YELLOW, GREEN, RED, ORANGE, VIOLET_BACK, BOLD, NC

# --- UI STYLE DEFINITIONS ---
# Defines the color scheme and formatting for the InquirerPy interactive prompts
MENU_STYLE = get_style(
    {
        'questionmark': '#004d4d bold underline',
        'answermark': 'magenta',
        'answer': 'cyan',
        "input": "#98c379",
        "question": "#004d4d bold underline",
        "answered_question": "magenta",
        "pointer": "fg: #f08a04 bold"
    }
)


class CaptionerCLI:
    """
    The interactive terminal interface for the AI Image Captioner.
    
    Manages user interactions, configuration updates, and pipeline execution
    through a series of hierarchical, asynchronous menus.
    """

    def __init__(self):
        """
        Initializes the CLI instance.
        
        The underlying `Captioner` instance is set to None initially and 
        instantiated when the CLI starts.
        """
        self.captioner: Captioner | None = None

    async def start(self) -> None:
        """
        Bootstraps the CLI interface.
        
        Clears the terminal, prints the stylized ASCII art banner, initializes 
        the core orchestrator, and enters the main menu loop.
        """
        print("\033c\n", end="")  # Clear terminal
        print(
            f"{CYAN}{BOLD}{VIOLET_BACK}" + " " * 67 + "\n" + f"{NC}"
            f"{CYAN}" + Figlet(font="thick", width=100).renderText(" AI Captioner\n") +
            f"{ORANGE}" + f"{VIOLET_BACK}" + " " * 67 +
            f"{NC}" + "\n"
        )
        self.captioner = Captioner()
        await self.main_menu()

    # --- INPUT WRAPPERS ---

    async def _handle_return(self, out: any, forward_to: Callable | tuple | None) -> bool:
        """
        Internal helper: intercepts user requests to go back or quit.
        
        If the user input matches a quit/back trigger, it jumps to the designated 
        `forward_to` coroutine and indicates the action was taken.

        Args:
            out: The raw output captured from the inquirer prompt.
            forward_to: The target coroutine or tuple representing the menu to return to.
                May be `None` if the caller wishes to handle navigation manually.

        Returns:
            bool: True if a return/back action was triggered and handled, False otherwise.
        """
        if out in ('q', 'Back', False):
            if forward_to:
                if isinstance(forward_to, tuple):
                    await forward_to[0](*forward_to[1:])
                else:
                    await forward_to()
            return True
        return False

    async def _basic_prompt(self, prompt_coro: any, forward_to: Callable | tuple | None) -> any:
        """
        Executes an InquirerPy coroutine while automatically handling back/quit logic.

        Args:
            prompt_coro: The inquirer object factory (e.g., `inquirer.text(...)`) 
                prior to calling `execute_async()`.
            forward_to: The target navigation point if the user backs out.

        Returns:
            any: The validated user input, or None if the user chose to back out.
        """
        out = await prompt_coro.execute_async()
        if await self._handle_return(out, forward_to):
            return None
        return out

    async def require_int(
            self,
            message: str,
            forward_to: Callable[..., Coroutine] | tuple[Callable[..., Coroutine], ...] | None,
            validate=None
        ) -> int | None:
        """
        Prompts the user for a strictly integer value.

        Args:
            message: The prompt text to display to the user.
            forward_to: Where to route the user if they cancel the prompt.
            validate: Optional validation logic for the input.

        Returns:
            int | None: The provided integer, or None if cancelled.
        """
        out = await self._basic_prompt(
            inquirer.number(message=message, style=MENU_STYLE, validate=validate),
            forward_to
        )
        return int(out) if out is not None else None

    async def require_float(
            self,
            message: str,
            forward_to: Callable[..., Coroutine] | tuple[Callable[..., Coroutine], ...] | None,
            validate=None
        ) -> float | None:
        """
        Prompts the user for a floating-point number.

        Args:
            message: The prompt text to display to the user.
            forward_to: Where to route the user if they cancel the prompt.
            validate: Optional validation logic for the input.

        Returns:
            float | None: The provided float, or None if cancelled.
        """
        out = await self._basic_prompt(
            inquirer.number(message=message, float_allowed=True, style=MENU_STYLE, validate=validate),
            forward_to
        )
        return float(out) if out is not None else None
        
    async def require_selection(
            self,
            message: str,
            choices: list[str],
            forward_to: Callable[..., Coroutine] | tuple[Callable[..., Coroutine], ...] | None
        ) -> str | None:
        """
        Displays an interactive selection list to the user.
        
        Automatically appends a "Back" option to the provided choices list.

        Args:
            message: The prompt text to display.
            choices: A list of string options for the user to select.
            forward_to: Where to route the user if they cancel or select "Back".

        Returns:
            str | None: The selected string, or None if cancelled.
        """
        out = await self._basic_prompt(
            inquirer.select(message=message, choices=choices + ["Back"], style=MENU_STYLE),
            forward_to
        )
        return out

    async def require_text(
            self,
            message: str,
            forward_to: Callable[..., Coroutine] | tuple[Callable[..., Coroutine], ...] | None            
        ) -> str | None:
        """
        Prompts the user to enter arbitrary text input.

        Args:
            message: The prompt text to display to the user.
            forward_to: Where to route the user if they cancel the prompt.

        Returns:
            str | None: The entered text, or None if cancelled.
        """
        out = await self._basic_prompt(
            inquirer.text(message=message, style=MENU_STYLE),
            forward_to
        )
        return out

    # --- MENUS ---

    async def main_menu(self) -> None:
        """
        Renders the root navigation menu for the application.
        
        Provides access to pipeline execution, cost estimation, and configurations.
        """
        while True:
            choice = await inquirer.select(
                message="Main menu:",
                choices=[
                    "🚀 Run captioning pipeline",
                    "💰 Evaluate expected cost",
                    "⚙️  Options menu",
                    "⛔ Close"
                ],
                style=MENU_STYLE
            ).execute_async()

            match choice:
                case "🚀 Run captioning pipeline":
                    await self.run_pipeline()

                case "💰 Evaluate expected cost":
                    await self.evaluate_costs()
                
                case "⚙️  Options menu":
                    await self.options_menu()

                case "⛔ Close":
                    sys.exit()

    async def run_pipeline(self) -> None:
        """
        Validates prerequisites and initiates the core asynchronous captioning pipeline.
        
        Checks for the presence of the active provider's API key before execution.
        """
        if self.captioner.config['api']['providers'][self.captioner.config["api"]["active"]["provider"]]['api_key'] == "":
            print(f"{RED}❌ API key is missing for the active provider: {BOLD}{self.captioner.config['api']['active']['provider']}{NC}{RED}!{NC}\n")
            return
            
        try:
            await self.captioner.run()            
        except Exception as e:
            print(f"\n{RED}🚨 Pipeline Error: {str(e)}{NC}\n")

    async def evaluate_costs(self) -> None:
        """
        Executes the CostEvaluator and prints projected token usage and USD expenses
        based on the current dataset and active model pricing.
        """
        results = self.captioner.cost_evaluator.calculate()
        
        if "error" in results:
            print(f"{RED}🚨 {results['error']}{NC}\n")
        else:
            print(f"\n{YELLOW}--- {self.captioner.config['api']['active']['model'].upper()} cost analysis ---{NC}")
            print(f"🌄 {CYAN}Pending images: {BOLD}{results['images']}{NC}")
            print(f"🎟️  {CYAN}Est. tokens:    {BOLD}{results['tokens']['total']:,} total{NC}")
            print(f"💲 {CYAN}Est. cost:      {BOLD}${results['usd']['total']:.4f} USD{NC}\n")

    async def options_menu(self) -> None:
        """
        Renders the primary configuration hub.
        
        Allows the user to view current settings, launch the prompt editor, 
        or navigate to deeper configuration sub-menus.
        """
        while True:
            choice = await self.require_selection(
                message="Options menu:",
                choices=[
                    "See current options",
                    "See prompt",
                    "App options",
                    "AI provider options",
                    "LoRA rules options"
                ],
                forward_to=self.main_menu
            )

            if not choice:
                return
            
            match choice:
                case "See current options":
                    # Extract configurations for formatted display
                    config = self.captioner.config
        
                    app_cfg = config.get('app', {})
                    lora_cfg = config.get('lora', {})
                    api_cfg = config.get('api', {})
                    active_api = api_cfg.get('active', {})
                    providers = api_cfg.get('providers', {})

                    print(f"\n{YELLOW}{BOLD}--- Current configuration ---{NC}")
                    
                    # App Options Output
                    print(f"\n{CYAN}{BOLD}[App Options]{NC}")
                    print(f"  Dataset Directory       : {app_cfg.get('dataset_dir')}")
                    print(f"  Max Concurrent Requests : {app_cfg.get('max_concurrent_requests')}")
                    print(f"  Max Image Size          : {app_cfg.get('max_image_size')}")
                    print(f"  Max Retries             : {app_cfg.get('max_retries')}")

                    # LoRA Rules Output
                    print(f"\n{CYAN}{BOLD}[LoRA Rules]{NC}")
                    print(f"  Trigger Word            : {lora_cfg.get('trigger_word')}")
                    print(f"  Min Caption Words       : {lora_cfg.get('min_words')}")
                    print(f"  Max Caption Words       : {lora_cfg.get('max_words')}")

                    # AI Settings Output
                    print(f"\n{CYAN}{BOLD}[AI Settings]{NC}")
                    print(f"  Active Provider         : {active_api.get('provider')}")
                    print(f"  Active Model            : {active_api.get('model')}")
                    print(f"  Max Tokens              : {api_cfg.get('max_tokens')}")
                    print(f"  Temperature             : {api_cfg.get('temperature')}")

                    # API Keys Security Audit Output
                    print(f"\n{CYAN}{BOLD}[API Keys Status]{NC}")
                    for prov_name, prov_data in providers.items():
                        key = prov_data.get('api_key', '')
                        # Safely check if the key exists and isn't a blank string
                        status = f"{GREEN}set{NC}" if key and str(key).strip() else f"{RED}missing{NC}"
                        print(f"  {prov_name.capitalize():<24}: {status}")
                        
                    print(f"\n{YELLOW}-----------------------------{NC}\n")
                
                case "See prompt":
                    # Attempt to invoke the OS-level default text editor for the prompt file
                    try:
                        if platform.system() == 'Darwin':       # macOS
                            subprocess.call(('open', self.captioner.prompt_path))
                        elif platform.system() == 'Windows':    # Windows
                            os.startfile(self.captioner.prompt_path)
                        else:                                   # Linux variants
                            subprocess.call(('xdg-open', self.captioner.prompt_path))
                        print(f"{GREEN}✅ Opened prompt file in default editor.{NC}\n")
                    except Exception as e:
                        print(f"{RED}🚨 Could not open editor: {e}{NC}\n")

                case "App options":
                    await self.app_options_menu()
                
                case "AI provider options":
                    await self.ai_client_menu()   
                
                case "LoRA rules options":
                    await self.lora_rules_menu()

    async def app_options_menu(self) -> None:
        """
        Renders the sub-menu for general application behavioral settings.
        
        Handles updates to dataset location, concurrency limits, and image bounds.
        """
        app_config = self.captioner.config.get('app')
        
        while True:
            choice = await self.require_selection(
                message=f"Configure app options:",
                choices=[
                    "Set dataset directory",
                    "Set maximum concurrent requests",
                    "Set maximum image size",
                    "Set maximum retries"
                ],
                forward_to=self.options_menu
            )
            
            match choice:
                case "Set dataset directory":
                    dataset_dir = await self.require_text(
                        message="Insert dataset directory path (e.g., ./dataset/raw_images):",
                        forward_to=self.app_options_menu
                    )
                    if dataset_dir is None:
                        return
                    else:
                        self.captioner.config['app']['dataset_dir'] = dataset_dir

                case "Set maximum concurrent requests":
                    max_concurrent_requests = await self.require_int(
                        message="Select maximum concurrent requests (API parallelization limit):",
                        forward_to=self.app_options_menu
                    )
                    if max_concurrent_requests is None:
                        return
                    else:
                        self.captioner.config['app']['max_concurrent_requests'] = max_concurrent_requests

                case "Set maximum image size":
                    max_image_size = await self.require_int(
                        message="Select maximum image size (downscaling target for API efficiency):",
                        forward_to=self.app_options_menu
                    )
                    if max_image_size is None:
                        return
                    else:
                        self.captioner.config['app']['max_image_size'] = max_image_size

                case "Set maximum retries":
                    max_retries = await self.require_int(
                        message="Select maximum retries;",
                        forward_to=self.app_options_menu
                    )
                    if max_retries is None:
                        return
                    else:
                        self.captioner.config['app']['max_retries'] = max_retries

    async def ai_client_menu(self) -> None:
        """
        Renders the sub-menu for managing AI providers and API configurations.
        
        Handles active provider routing, model selection, API key injection, 
        and generation limits (tokens/temperature).
        """
        api_config = self.captioner.config.get('api')
        while True:
            choice = await self.require_selection(
                message=f"Configure AI client:",
                choices=[
                    "Select provider",
                    "Select model",
                    "Set API key",
                    "Select maximum tokens",
                    "Select temperature"
                ],
                forward_to=self.options_menu
            )
            
            match choice:
                case "Select provider":
                    provider = await self.require_selection(
                        message="Select AI provider:",
                        choices=list(api_config.get("providers").keys()),
                        forward_to=self.ai_client_menu
                    )
                    if not provider:
                        return
                    else:
                        self.captioner.config['api']['active']['provider'] = provider

                    # Automatically prompt for model selection after changing provider
                    model = await self.require_selection(
                        message=f"Select AI model for {api_config.get('active').get('provider')} provider:",
                        choices=list(api_config.get("providers").get(api_config.get('active').get('provider')).get('models')),
                        forward_to=self.ai_client_menu
                    )
                    if not model:
                        return
                    else:
                        self.captioner.config['api']['active']['model'] = model

                case "Select model":
                    model = await self.require_selection(
                        message=f"Select AI model for {api_config.get('active').get('provider')} provider:",
                        choices=list(api_config.get("providers").get(api_config.get('active').get('provider')).get('models')),
                        forward_to=self.ai_client_menu
                    )
                    if not model:
                        return
                    else:
                        self.captioner.config['api']['active']['model'] = model

                case "Set API key":
                    api_key = await self.require_text(
                        message=f"Insert API key for {api_config.get('active').get('provider')} provider:",
                        forward_to=self.ai_client_menu
                    )
                    if api_key is None:
                        return
                    else:
                        self.captioner.config['api']['providers'][api_config.get('active').get('provider')]['api_key'] = api_key

                case "Select maximum tokens":
                    max_tokens = await self.require_int(
                        message="Select maximum tokens (length of caption):",
                        forward_to=self.ai_client_menu
                    )
                    if max_tokens is None:
                        return
                    else:
                        self.captioner.config['api']['max_tokens'] = max_tokens

                case "Select temperature":
                    temperature = await self.require_float(
                        message="Select temperature:",
                        forward_to=self.ai_client_menu
                    )
                    if temperature is None:
                        return
                    else:
                        self.captioner.config['api']['temperature'] = temperature

            # Persist changes to config.yaml after any mutation in this menu
            self.captioner._save_config()
       
    async def lora_rules_menu(self) -> None:       
        """
        Renders the sub-menu for defining LoRA-specific captioning boundaries.
        
        Handles the trigger word injection and output density limits (min/max words).
        """
        while True:
            choice = await self.require_selection(
                message=f"Configure LoRA rules:",
                choices=[
                    "Set trigger word",
                    "Set min caption words",
                    "Set max caption words"
                ],
                forward_to=self.options_menu
            )
            
            match choice:
                case "Set trigger word":
                    trigger_word = await self.require_text(
                        message="Insert trigger word:",
                        forward_to=self.lora_rules_menu
                    )
                    if trigger_word is None:
                        return
                    else:
                        self.captioner.config['lora']['trigger_word'] = trigger_word

                case "Set min caption words":
                    min_words = await self.require_int(
                        message="Insert caption min words:",
                        forward_to=self.lora_rules_menu
                    )
                    if min_words is None:
                        return
                    else:
                        self.captioner.config['lora']['min_words'] = min_words

                case "Set max caption words":
                    max_words = await self.require_int(
                        message="Insert caption max words:",
                        forward_to=self.lora_rules_menu
                    )
                    if max_words is None:
                        return
                    else:
                        self.captioner.config['lora']['max_words'] = max_words

            # Persist changes to config.yaml after any mutation in this menu
            self.captioner._save_config()