"""
Utility functions for PyEcharts chart rendering in PPT.
This module handles chart snapshot generation using Playwright.
"""

import asyncio
import logging
import os
import threading
import time

import requests


def _get_font_dir():
    """Get the directory for storing font files."""
    # Store fonts in the same directory as this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(module_dir, 'fonts')
    os.makedirs(font_dir, exist_ok=True)
    return font_dir


def _ensure_chinese_font():
    """
    Ensure a Chinese font file is available locally.
    Downloads Noto Sans SC if not present.
    
    Returns:
        str: Path to the font file, or None if unavailable
    """
    font_dir = _get_font_dir()
    font_path = os.path.join(font_dir, 'NotoSansSC-Regular.ttf')
    
    if os.path.exists(font_path):
        return font_path
    
    # Try to download the font
    font_urls = [
        # Mirror sources for better availability
        'https://cdn.jsdelivr.net/gh/notofonts/noto-cjk/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',
        'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',
    ]
    
    for url in font_urls:
        try:
            logging.info(f"Downloading Chinese font from {url}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Chinese font downloaded to {font_path}")
                return font_path
        except Exception as e:
            logging.warning(f"Failed to download font from {url}: {e}")
            continue
    
    logging.warning("Could not download Chinese font. Text may not display correctly.")
    return None


def inject_chinese_font_into_html(html_path: str, font_family: str):
    """
    Inject Chinese web font CSS into the PyEcharts HTML file to ensure
    Chinese characters display correctly in headless browsers.
    
    Args:
        html_path: Path to the HTML file
        font_family: Font family string to use
    """
    # First try to use local font file
    local_font_path = _ensure_chinese_font()
    
    if local_font_path and os.path.exists(local_font_path):
        # Use local font file with file:// protocol
        abs_font_path = os.path.abspath(local_font_path)
        font_css = f'''
    <style>
        @font-face {{
            font-family: 'Noto Sans SC';
            font-style: normal;
            font-weight: 400;
            font-display: block;
            src: url('file://{abs_font_path}') format('opentype');
        }}
        * {{
            font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', sans-serif !important;
        }}
        body, div, span, text, tspan {{
            font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', sans-serif !important;
        }}
    </style>
    '''
    else:
        # Fallback to web fonts with multiple CDN sources
        font_css = '''
    <style>
        /* Load Noto Sans SC from multiple sources for better reliability */
        @font-face {
            font-family: 'Noto Sans SC';
            font-style: normal;
            font-weight: 400;
            font-display: swap;
            src: local('Noto Sans SC Regular'), local('NotoSansSC-Regular'),
                 url('https://cdn.staticfile.net/lxgw-wenkai-webfont/1.7.0/style.css') format('woff2'),
                 url('https://fonts.gstatic.com/s/notosanssc/v36/k3kCo84MPvpLmixcA63oeAL7Iqp5IZJF9bmaG9_FnYxNbPzS5HE.119.woff2') format('woff2');
        }
        @import url('https://fonts.loli.net/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
        
        /* Apply font to all elements */
        * {
            font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', sans-serif !important;
        }
        body, div, span, text, tspan {
            font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', sans-serif !important;
        }
    </style>
    '''
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject the font CSS right after <head> tag
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', '<head>' + font_css)
        elif '<HEAD>' in html_content:
            html_content = html_content.replace('<HEAD>', '<HEAD>' + font_css)
        else:
            # Fallback: prepend to the beginning
            html_content = font_css + html_content
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logging.info(f"Injected Chinese font CSS into {html_path}")
    except Exception as e:
        logging.warning(f"Failed to inject font CSS: {e}")


def check_playwright_available():
    """
    Check if Playwright is available and properly configured.
    
    Returns:
        tuple: (is_available: bool, error_message: str or None)
    """
    try:
        import playwright
        return True, None
    except ImportError:
        return False, (
            "Playwright is not installed. "
            "Please install with: pip install playwright && playwright install chromium"
        )


async def _playwright_async_snapshot(html_path: str, output_path: str):
    """Use Playwright Async API to render chart to image with transparent background."""
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        # Launch with additional args for better compatibility on various Linux systems
        browser = await p.chromium.launch(
            headless=True,
            timeout=60000,  # Increase launch timeout to 60 seconds
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--single-process',  # May help with stability in some environments
                '--disable-extensions',
                '--disable-background-networking',
            ]
        )
        try:
            page = await browser.new_page()
            
            # Convert to absolute path and file URL
            abs_path = os.path.abspath(html_path)
            file_url = f"file://{abs_path}"
            
            await page.goto(file_url, wait_until='networkidle', timeout=30000)
            
            # Wait for fonts to load
            try:
                await page.wait_for_function('''
                    () => document.fonts && document.fonts.ready
                ''', timeout=5000)
            except Exception:
                pass  # Fallback if fonts API not available
            
            # Additional wait for chart and fonts to render
            await page.wait_for_timeout(2500)
            
            # Try to find the chart container element and screenshot only that element
            # PyEcharts renders chart inside a div with specific class or the canvas element
            chart_element = await page.query_selector('div[_echarts_instance_]')
            if chart_element is None:
                chart_element = await page.query_selector('canvas')
            if chart_element is None:
                chart_element = await page.query_selector('#main')
            if chart_element is None:
                # Fallback to the first div with width/height style (chart container)
                chart_element = await page.query_selector('div[style*="width"][style*="height"]')
            
            if chart_element:
                # Screenshot only the chart element with transparent background
                await chart_element.screenshot(path=output_path, omit_background=True)
            else:
                # Fallback: screenshot the full page but clip to viewport
                await page.screenshot(path=output_path, full_page=False, omit_background=True)
        finally:
            await browser.close()


def playwright_snapshot(html_path: str, output_path: str, max_retries: int = 3):
    """
    Use Playwright to render chart to image with transparent background.
    Automatically handles async context by running in a separate thread.
    
    Args:
        html_path: Path to the HTML file containing the chart
        output_path: Path where the screenshot image will be saved
        max_retries: Maximum number of retry attempts (default: 3)
    """
    last_exception = None
    
    # First try: async API in separate thread (original approach)
    for attempt in range(max_retries):
        exception = [None]
        
        def run_in_thread():
            try:
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(_playwright_async_snapshot(html_path, output_path))
                finally:
                    new_loop.close()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=120)  # Add timeout to prevent hanging
        
        if thread.is_alive():
            logging.warning(f"Playwright async snapshot timed out on attempt {attempt + 1}")
            last_exception = TimeoutError("Playwright snapshot timed out")
            continue
        
        if exception[0] is None:
            # Success
            return
        
        last_exception = exception[0]
        logging.warning(f"Playwright async snapshot failed on attempt {attempt + 1}: {exception[0]}")
        
        if attempt < max_retries - 1:
            time.sleep(1)  # Brief delay before retry
    
    # All attempts failed
    raise last_exception
