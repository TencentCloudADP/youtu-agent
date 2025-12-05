"""
Image generation toolkit using Doubao SeeDream API
https://ark.cn-beijing.volces.com/api/v3/images/generations
"""

import os
import re
import base64
import tempfile
from typing import Union, List, Optional
from pathlib import Path
import aiohttp
import aiofiles
from PIL import Image
import io

from ..config import ToolkitConfig
from ..utils import get_logger, SimplifiedAsyncOpenAI
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)

# Precompiled regex pattern for parsing optimized prompt
REVISED_PROMPT_PATTERN = re.compile(r'<revised_prompt>(.*?)</revised_prompt>', re.DOTALL)

# Mode mapping for image generation
MODE_MAPPING = {
    "single": "disabled",  # Generate single image
    "multiple": "auto"     # Generate multiple images
}

# Prompt optimization instruction (adjustable)
PROMPT_OPTIMIZATION_INSTRUCTION = """You are a professional AI image generation prompt optimization expert for Seedream 4.0. Enhance the user's prompt to create more detailed, specific, and expressive descriptions for higher quality image generation.

Optimization Guidelines:
1. Use clear, natural language to describe: Subject + Action + Environment
2. Add aesthetic elements naturally: style, color, lighting, composition
3. For specific use cases, explicitly state the image purpose and type
   Example: "Design a game company logo featuring a dog playing with a gamepad, with 'PITBULL' written on it"
4. For text rendering, enclose text content in double quotes
   Example: 'A poster with title "Seedream 4.0"'
5. For style requirements, use precise style keywords or descriptions
6. Keep prompts concise (under 300 Chinese characters or 600 English words) - excessive details may cause the model to miss elements
7. **CRITICAL: Preserve the original language strictly** - If the original prompt is in Chinese, the revised prompt MUST be in Chinese; if English, MUST be in English. Do NOT translate or change the language.
8. Maintain the original intent while enhancing clarity and detail

Good Examples:
- "A girl in gorgeous attire walking down a tree-lined avenue with a parasol, in Monet oil painting style"
- "A messy office desk with an open laptop displaying green code, a mug labeled 'Developer' with steam, an open book showing a Venn diagram, and sunlight casting shadows from the right"

Avoid:
- Fragmented phrases: "a girl, umbrella, tree-lined street, oil painting-like delicate brushstrokes"
- Vague references without context
- Overly complex, repetitive descriptions

User's original prompt: {original_prompt}

Please place the optimized prompt inside <revised_prompt> tags:
<revised_prompt>Your optimized prompt here</revised_prompt>

Do not add any other explanations or extra content."""


class ImageGenToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.api_key = os.getenv("ARK_API_KEY")
        self.model = os.getenv("ARK_MODEL", "doubao-seedream-4-0-250828")
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        self.temp_dir = tempfile.gettempdir()
        # Initialize LLM for prompt optimization
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
    
    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        return path.startswith(('http://', 'https://'))
    
    def _validate_and_compress_image(self, image_path: str) -> str:
        """Validate and compress image if necessary.
        
        Requirements:
        - Format: jpeg, png
        - Aspect ratio (width/height): [1/3, 3]
        - Width and height (px) > 14
        - Size: <= 10MB
        - Total pixels: <= 6000×6000 px
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Check minimum dimensions
            if width <= 14 or height <= 14:
                raise ValueError(f"Image dimensions too small: {width}x{height}. Must be > 14px")
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 1/3 or aspect_ratio > 3:
                logger.warning(f"Aspect ratio {aspect_ratio:.2f} out of range [0.33, 3.0]. Adjusting...")
                # Crop to fit aspect ratio
                if aspect_ratio < 1/3:
                    new_height = int(width * 3)
                    img = img.crop((0, (height - new_height) // 2, width, (height + new_height) // 2))
                else:
                    new_width = int(height * 3)
                    img = img.crop(((width - new_width) // 2, 0, (width + new_width) // 2, height))
                width, height = img.size
            
            # Check total pixels
            total_pixels = width * height
            if total_pixels > 6000 * 6000:
                logger.warning(f"Total pixels {total_pixels} exceeds 36M. Resizing...")
                scale = (6000 * 6000 / total_pixels) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to RGB if necessary (for PNG with alpha channel)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save and check file size
            output_path = os.path.join(self.temp_dir, f"compressed_{os.path.basename(image_path)}")
            quality = 95
            while quality > 10:
                img.save(output_path, format='JPEG', quality=quality, optimize=True)
                file_size = os.path.getsize(output_path)
                if file_size <= 10 * 1024 * 1024:  # 10MB
                    break
                quality -= 10
            
            if os.path.getsize(output_path) > 10 * 1024 * 1024:
                raise ValueError(f"Unable to compress image to under 10MB")
            
            logger.info(f"Image validated and compressed: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f}MB)")
            return output_path
        
        except Exception as e:
            logger.error(f"Error validating/compressing image {image_path}: {str(e)}")
            raise
    
    async def _download_image(self, url: str) -> str:
        """Download image from URL to temporary directory."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image from {url}. Status: {response.status}")
                    
                    # Generate temp file name
                    file_ext = url.split('.')[-1].split('?')[0]
                    if file_ext not in ['jpg', 'jpeg', 'png']:
                        file_ext = 'jpg'
                    temp_path = os.path.join(self.temp_dir, f"downloaded_{hash(url)}_{os.urandom(4).hex()}.{file_ext}")
                    
                    # Save downloaded image
                    async with aiofiles.open(temp_path, 'wb') as f:
                        await f.write(await response.read())
                    
                    logger.info(f"Downloaded image from {url} to {temp_path}")
                    return temp_path
        
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            raise
    
    async def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string with proper data URI format.
        
        Returns base64 string in format: data:image/<format>;base64,<base64_data>
        where <format> is in lowercase (e.g., png, jpeg)
        """
        try:
            async with aiofiles.open(image_path, 'rb') as f:
                image_data = await f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                
                # Detect image format
                img = Image.open(io.BytesIO(image_data))
                image_format = img.format.lower() if img.format else 'jpeg'
                
                # Normalize format names
                if image_format == 'jpg':
                    image_format = 'jpeg'
                
                # Return base64 with proper data URI prefix
                return f"data:image/{image_format};base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise
    
    async def _process_images(self, images: Union[str, List[str]]) -> List[str]:
        """Process images: download if URL, validate, compress, and convert to base64.
        
        Args:
            images: Single image path/URL or list of image paths/URLs (max 10)
        
        Returns:
            List of base64 encoded image strings
        """
        if isinstance(images, str):
            images = [images]
        
        if len(images) > 10:
            raise ValueError(f"Too many reference images: {len(images)}. Maximum is 10.")
        
        base64_images = []
        temp_files = []
        
        try:
            for img in images:
                # Download if URL
                if self._is_url(img):
                    local_path = await self._download_image(img)
                    temp_files.append(local_path)
                else:
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    local_path = img
                
                # Validate and compress
                processed_path = self._validate_and_compress_image(local_path)
                if processed_path != local_path:
                    temp_files.append(processed_path)
                
                # Convert to base64
                base64_str = await self._image_to_base64(processed_path)
                base64_images.append(base64_str)
            
            return base64_images
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {str(e)}")
    
    async def _optimize_prompt(self, prompt: str) -> str:
        """Optimize the prompt using LLM to make it more detailed and expressive.
        
        Args:
            prompt (str): Original prompt to optimize
            
        Returns:
            str: Optimized prompt
        """
        try:
            instruction = PROMPT_OPTIMIZATION_INSTRUCTION.format(original_prompt=prompt)
            response = await self.llm.query_one(
                messages=[{"role": "user", "content": instruction}],
                **self.config.config_llm.model_params.model_dump() if self.config.config_llm else {}
            )
            
            # Parse the response to extract content from <revised_prompt> tags
            match = REVISED_PROMPT_PATTERN.search(response)
            if match:
                optimized = match.group(1).strip()
                logger.info(f"Optimized prompt: {optimized}")
                return optimized
            else:
                logger.warning(f"Could not find <revised_prompt> tags in response. Using response as-is: {response}")
                return response.strip()
        except Exception as e:
            logger.error(f"Failed to optimize prompt: {str(e)}. Using original prompt.")
            return prompt

    @register_tool
    async def generate_image(self, prompt: str, ref_image: Optional[Union[str, List[str]]] = None, save_path: str = None, mode: str = "single", auto_enhance_prompt: bool = False) -> str:
        r"""Generates an image based on the given prompt and optional reference images, then saves it to the specified path.

        Args:
            prompt (str): The text prompt describing the image to generate.
            ref_image (str or list of str, optional): Reference image(s) as local path(s) or URL(s). 
                Maximum 10 images. Images will be validated, compressed if needed, and converted to base64.
            save_path (str): The local file path where the generated image will be saved.
                For multiple images, files will be saved with suffixes (_0, _1, _2, etc.).
            mode (str, optional): Generation mode - "single" for single image, "multiple" for multiple images. 
                Default is "single".
            auto_enhance_prompt (bool, optional): Additional feature to automatically enhance and optimize 
                the prompt using LLM before image generation. The LLM will add visual details, artistic 
                terminology, and improve the overall quality of the prompt for better image results. 
                Default is True.
        """
        # Auto-enhance prompt if enabled
        if auto_enhance_prompt:
            logger.info(f"Original prompt: {prompt}")
            prompt = await self._optimize_prompt(prompt)
            logger.info(f"Using enhanced prompt: {prompt}")
        
        # Validate and map mode parameter
        if mode not in MODE_MAPPING:
            logger.warning(f"Invalid mode '{mode}'. Valid options are: {list(MODE_MAPPING.keys())}. Using 'single' as default.")
            mode = "single"
        
        api_mode = MODE_MAPPING[mode]
        logger.info(f"Generation mode: {mode} (API mode: {api_mode})")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "url",
            "sequential_image_generation": api_mode,
            "size": "2k",
            "watermark": False
        }
        
        # Process reference images if provided
        if ref_image:
            try:
                base64_images = await self._process_images(ref_image)
                payload["image"] = base64_images
                logger.info(f"Added {len(base64_images)} reference image(s) to the request")
            except Exception as e:
                logger.error(f"Failed to process reference images: {str(e)}")
                return f"Error: Failed to process reference images - {str(e)}"

        try:
            async with aiohttp.ClientSession() as session:
                # Call the image generation API
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return f"Error: Failed to generate image. Status code: {response.status}"
                    
                    result = await response.json()
                    logger.debug(f"Image generation response: {result}")
                    
                    # Extract the image URLs from the response
                    if "data" not in result or len(result["data"]) == 0:
                        logger.error("No image data in API response")
                        return "Error: No image data returned from API"
                    
                    # Handle save_path: use default if not provided
                    if not save_path:
                        save_path = f"{prompt}.jpg"
                        logger.info(f"No save_path provided, using default: {save_path}")
                    
                    # Ensure the directory exists (if save_path contains directory)
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    
                    # Get base path and extension for naming multiple images
                    base_path = os.path.splitext(save_path)[0]
                    extension = os.path.splitext(save_path)[1] or ".jpg"
                    
                    saved_files = []
                    
                    # Download all generated images
                    for idx, image_data in enumerate(result["data"]):
                        image_url = image_data["url"]
                        image_size = image_data.get("size", "unknown")
                        
                        # Construct file path with suffix for multiple images
                        if len(result["data"]) > 1:
                            current_save_path = f"{base_path}_{idx}{extension}"
                        else:
                            current_save_path = save_path
                        
                        # Download the image from the URL
                        async with session.get(image_url) as img_response:
                            if img_response.status != 200:
                                logger.error(f"Failed to download image {idx} from {image_url}")
                                continue
                            
                            # Save the image to the specified path
                            async with aiofiles.open(current_save_path, 'wb') as f:
                                await f.write(await img_response.read())
                        
                        saved_files.append(f"{current_save_path} (size: {image_size})")
                        logger.info(f"Image {idx} successfully saved to {current_save_path} (size: {image_size})")
                    
                    if not saved_files:
                        return "Error: Failed to download any generated images"
                    
                    # Return success message with all saved file paths
                    if len(saved_files) == 1:
                        return f"Image generated successfully and saved to {saved_files[0]}"
                    else:
                        files_list = "\n".join([f"  - {f}" for f in saved_files])
                        return f"Generated {len(saved_files)} images successfully:\n{files_list}"
        
        except aiohttp.ClientError as e:
            logger.error(f"Network error during image generation: {str(e)}")
            return f"Error: Network error - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during image generation: {str(e)}")
            return f"Error: {str(e)}"
