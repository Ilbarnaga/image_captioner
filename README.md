AI CAPTIONER - ADVANCED LoRA DATASET PIPELINE
================================================================================

1. OVERVIEW
--------------------------------------------------------------------------------
AI Captioner is an elite, asynchronous Vision-Language pipeline designed specifically for generating highly optimized, natural language dataset captions. It is tailor-made for training next-generation Diffusion Transformer (DiT) LoRAs—such as Z-Image, Wan-T2I, Qwen-Image, and Flux—which utilize massive Large Language Models (like T5 or Qwen2.5) as their text encoders.

Unlike legacy tools that generate disconnected, comma-separated "Booru" tags for CLIP models (e.g., SD 1.5), AI Captioner generates dense, grammatically correct prose. It utilizes a strict "subtractive prompting" logic and an automated Quality Control (QC) bouncer to ensure your LoRA learns the character's identity perfectly without concept bleed.


2. KEY FEATURES
--------------------------------------------------------------------------------
* Asynchronous Pipeline: Processes multiple images concurrently using asyncio to maximize API throughput and reduce dataset prep time.
* Intelligent Quality Control (QC): A built-in bouncer that validates every generated caption. It enforces word-count boundaries, ensures trigger word placement, strips out AI conversational filler ("Here is a picture of..."), and bans legacy comma-separated tags.
* Auto-Correcting Retry Loop: If a caption fails the QC check, the application automatically appends the failure reason to the prompt and forces the Vision AI to rewrite and fix the caption.
* Cost Evaluator: Before running a massive dataset, calculate your exact estimated API costs (in USD) and token usage based on the active model's specific vision tokenization formulas.
* Interactive CLI: A beautiful, terminal-based UI (powered by InquirerPy) that allows you to configure rules, swap AI providers, and edit prompts without ever touching the code.
* Native Checkpointing: If the process is interrupted, restarting it will safely skip any images that already have a corresponding .txt file.


3. SUPPORTED AI PROVIDERS
--------------------------------------------------------------------------------
The application supports the top-tier Vision-Language APIs. You can hot-swap between these providers directly from the CLI.

* xAI (Grok)
  Highly recommended for cost-to-performance ratio. Uses models like grok-4-1-fast-non-reasoning.
  Website: https://x.ai/api
  Docs: https://docs.x.ai/docs

* OpenAI (GPT)
  The industry standard. Uses models like gpt-4o and the gpt-5.2 series.
  Website: https://platform.openai.com
  Docs: https://platform.openai.com/docs

* Anthropic (Claude)
  Exceptional spatial reasoning and natural language prose. Uses models like claude-3-5-sonnet-latest.
  Website: https://console.anthropic.com
  Docs: https://docs.anthropic.com


4. THE CLI INTERFACE (MAIN MENU)
--------------------------------------------------------------------------------
Upon launching the application, you are greeted with the Main Menu. Navigate using your keyboard arrows and Enter.

[🚀 Run captioning pipeline]
Starts the asynchronous processing loop. It scans your target dataset directory and begins hitting the active AI provider API to generate .txt files for every image.

[💰 Evaluate expected cost]
Scans your dataset to see how many unprocessed images remain. It calculates the base tokens required for the images, adds the prompt overhead, and calculates the maximum output tokens to give you a highly accurate USD cost estimate based on your active model.

[⚙️ Options menu]
Enters the configuration hub where you can tweak the application.

[⛔ Close]
Safely exits the application.


5. CONFIGURATION & OPTIONS MENU
--------------------------------------------------------------------------------
The Options Menu modifies the config/config.yaml file in real-time. 

[See current options]
Prints a formatted summary of your entire configuration, including which API keys are currently set or missing.

[See prompt]
Automatically detects your Operating System (Windows/Mac/Linux) and opens the core captioning_prompt.txt file in your system's default text editor.

[App options]
* Set dataset directory: The path where your raw images are stored (e.g., ./dataset).
* Set max concurrent requests: How many API calls to make at once (default: 5).
* Set max image size: The resolution images are downscaled to before being sent to the API to save tokens (default: 1024).
* Set max retries: How many times the AI can fail the QC check before the app gives up on that specific image (default: 3).

[AI provider options]
* Select provider: Hot-swap between openai, grok, and claude.
* Select model: Choose the specific model (e.g., gpt-4o, claude-3-5-sonnet).
* Set API key: Securely input your API key for the active provider.
* Select max tokens: The maximum length of the output caption (default: 300).
* Select temperature: Creativity/randomness of the AI (default: 0.3 for clinical accuracy).

[LoRA rules options]
* Set trigger word: The unique token representing your subject (e.g., elys1af0x).
* Set min caption words: Enforces density. If the AI writes fewer words than this, it is forced to retry (default: 80).
* Set max caption words: Prevents hallucination. If the AI writes more words than this, it is forced to retry (default: 120).


6. THE PROMPTING STRATEGY (LoRA BEST PRACTICES)
--------------------------------------------------------------------------------
The application relies on a .txt file located at src/prompting/captioning_prompt.txt. This prompt is heavily optimized for DiT architectures and photorealistic character training.

The "Golden Rule": Subtractive Prompting
The prompt strictly forbids the Vision AI from describing the character's permanent physical traits (eye color, face shape, jawline, permanent tattoos, etc.). By entirely omitting these features from the text captions, the DiT model is forced to associate those exact facial and bodily geometries strictly with your [TRIGGER_WORD]. 

If you describe her "blue eyes", the model will untangle the concept of blue eyes from your trigger word, resulting in a LoRA that forgets what your character looks like unless you prompt it perfectly. The AI is instead instructed to focus exhaustively on the flexible elements: clothing, pose, lighting, and background environments.


7. IMAGE PREPROCESSING OPTIMIZATIONS (THE JPEG-85 STANDARD)
--------------------------------------------------------------------------------
Under the hood, AI Captioner automatically processes and encodes your images using a strict 'JPEG format at 85 Quality' standard before sending them to the APIs.

Why JPEG?
* Payload Limits: APIs require Base64 encoding, which bloats file sizes by ~33%. A raw 4MB PNG becomes a 5.3MB payload, risking "413 Payload Too Large" errors.
* Alpha Stripping: PNGs and WebPs often carry transparency data (Alpha Channels). Vision-Language Models do not process transparency for dataset captioning. Forcing JPEG automatically strips this dead weight.
* Server Optimization: OpenAI, Anthropic, and xAI backend infrastructures decode standard RGB JPEGs significantly faster than other formats.

Why Quality 85?
* 85 is the universally accepted mathematical "Sweet Spot" for machine vision. 
* Above 85: File sizes increase exponentially with zero tangible benefit to the AI's visual comprehension, wasting bandwidth.
* Below 85: The compression algorithm introduces visible "macroblocking" (pixelation). If the AI encoder detects these artifacts, it may falsely inject destructive words like "low quality, pixelated, jpeg artifacts" into your LoRA captions.


8. QUICK START GUIDE
--------------------------------------------------------------------------------
1. Setup Environment:
   Ensure you have Python 3.10+ installed. Install required dependencies:
   pip install InquirerPy pyfiglet openai anthropic pyyaml colorama pillow

2. Prepare Dataset:
   Place all your .jpg, .png, or .webp training images into the ./dataset folder.

3. Launch Application:
   Run the CLI via terminal: python main.py (or your main entry point / .bat file).

4. Configure:
   Go to "Options -> AI provider options" to select your preferred model and paste your API key.
   Go to "Options -> LoRA rules options" to set your unique trigger word.

5. Evaluate & Run:
   Go back to the Main Menu. 
   Hit "Evaluate expected cost" to ensure your wallet is ready.
   Hit "Run captioning pipeline" and watch the magic happen.

The application will populate your dataset folder with .txt files matching your image names, perfectly formatted for Kohya_ss, OneTrainer, or any modern fine-tuning toolkit.
