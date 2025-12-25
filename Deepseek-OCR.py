"""
DeepSeek-OCR Complete Code with Model Loading
This script includes model initialization, fixed OCR extraction, and file saving.
"""

import sys
from io import StringIO
import os
import tempfile
import shutil
import glob
import json
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from google.colab import files
from IPython.display import display
import gradio as gr

print("✅ All imports loaded\n")

# ================================================================================
# LOAD MODEL (DeepSeek-OCR)
# ================================================================================

print("Step: Loading DeepSeek-OCR model...")
model_name = "deepseek-ai/DeepSeek-OCR"

try:
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    print(f"Loading model from {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='eager',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    model = model.eval()
    print("\n✅ Model loaded successfully!\n")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    # Continue anyway in case model is already loaded in Colab memory

# ================================================================================
# FIXED OCR FUNCTION - Captures stdout from model
# ================================================================================

def perform_ocr(image, prompt="<image>\nExtract all text from the image.", 
               base_size=1024, image_size=640, crop_mode=True):
    """
    ✅ FIXED: Captures the printed output from DeepSeek-OCR model
    The model prints text to stdout but returns None - this captures it!
    """
    try:
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            temp_image_path = tmp.name
        
        print(f"\n{'='*60}")
        print(f"Processing: {image.size} pixels")
        print(f"{'='*60}\n")
        
        temp_output = tempfile.mkdtemp()
        
        # ✅ KEY FIX: Capture what the model prints
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        
        try:
            # Run OCR
            model.infer(
                tokenizer=tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_output,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=False
            )
        finally:
            sys.stdout = old_stdout
            output_text = captured.getvalue()
        
        # Extract the actual content
        lines = output_text.split('\n')
        content = []
        collecting = False
        
        for line in lines:
            # Start collecting after technical output
            if 'SUMMARY' in line or 'SKILLS' in line or len(line) > 50:
                collecting = True
            # Stop at save results marker
            if 'save results' in line.lower():
                break
            
            if collecting:
                # Skip technical lines
                if not any(x in line for x in ['torch.Size', 'BASE:', 'PATCHES:', '====', 'image:', 'other:']):
                    content.append(line)
        
        result = '\n'.join(content).strip()
        
        # Cleanup
        os.unlink(temp_image_path)
        shutil.rmtree(temp_output, ignore_errors=True)
        
        if result and len(result) > 20:
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS! Extracted {len(result)} characters")
            print(f"{'='*60}")
            print("\nPreview (first 400 chars):")
            print(result[:400] + "..." if len(result) > 400 else result[:400])
            print(f"\n{'='*60}\n")
            return result
        else:
            print("⚠️ No text extracted")
            return ""
            
    except Exception as e:
        import traceback
        error = f"Error: {e}\n{traceback.format_exc()}"
        print(error)
        return error

print("✅ perform_ocr() function ready\n")

# ================================================================================
# GRADIO UI SETUP
# ================================================================================

def gradio_process_image(image):
    """Wrapper function for Gradio interface"""
    if image is None:
        return "Please upload an image.", None
    
    print("\nProcessing image via Gradio UI...")
    
    # Run OCR using the fixed perform_ocr function
    ocr_text = perform_ocr(image)
    
    # Save to a temporary file for download
    output_filename = "ocr_result.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(ocr_text)
        
    return ocr_text, output_filename

# Create Gradio Interface
print("\nLaunching Gradio UI...")
print("Click the public URL below to open the interface in a new tab.")

with gr.Blocks(title="DeepSeek-OCR Interface") as demo:
    gr.Markdown("# 👁️ DeepSeek-OCR Interface")
    gr.Markdown("Upload an image to extract text using the DeepSeek-OCR model loaded in this Colab runtime.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            process_btn = gr.Button("Extract Text", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Extracted Text", show_copy_button=True, lines=20)
            download_file = gr.File(label="Download Result (.txt)")
    
    process_btn.click(
        fn=gradio_process_image,
        inputs=[input_image],
        outputs=[output_text, download_file]
    )

# Launch the interface
# share=True creates a public link accessible from anywhere
demo.launch(share=True, debug=True)
