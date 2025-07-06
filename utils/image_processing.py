"""Image processing utilities for SitSmart application"""
import os
import re
import fitz
from PIL import Image
import streamlit as st

def load_table_transformer():
    """Load Table Transformer model for table detection"""
    try:
        import timm
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        return processor, model
    except ImportError:
        st.info("📋 Table Transformer not available. Using basic extraction. Install 'timm' for advanced table detection.")
        return None, None
    except Exception as e:
        st.warning(f"Could not load Table Transformer: {e}")
        return None, None

def detect_tables_in_page(page_image, processor, model):
    """Detect table regions in page image"""
    if processor is None or model is None:
        return []
    
    try:
        import torch
        inputs = processor(page_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([page_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:
                tables.append(box.tolist())
        
        return tables
    except Exception as e:
        print(f"Table detection error: {e}")
        return []

def extract_images_from_pdfs():
    """Extract images from PDFs with optional Table Transformer for better mapping"""
    images_data = {}
    if not os.path.exists("images"):
        os.makedirs("images")
    
    processor, model = load_table_transformer()
    use_table_transformer = processor is not None and model is not None
    
    for filename in os.listdir("documents"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("documents", filename)
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                if use_table_transformer:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    table_boxes = detect_tables_in_page(page_image, processor, model)
                    
                    text_dict = page.get_text("dict")
                    product_positions = []
                    
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    if re.match(r'SSC\s*-\s*[A-Z0-9]{2,4}', text, re.IGNORECASE):
                                        bbox = span["bbox"]
                                        product_positions.append({
                                            'name': text.upper().replace(' ', ''),
                                            'bbox': bbox
                                        })
                else:
                    page_text = page.get_text()
                    product_codes = re.findall(r'SSC\s*-\s*[A-Z0-9]{2,4}', page_text, re.IGNORECASE)
                    product_positions = [{'name': code.upper().replace(' ', ''), 'bbox': None} for code in product_codes]
                    table_boxes = []
                
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        
                        best_product = "Product"
                        
                        if use_table_transformer and table_boxes and product_positions:
                            if img_index < len(product_positions):
                                best_product = product_positions[img_index]['name']
                        elif product_positions:
                            best_product = product_positions[img_index % len(product_positions)]['name']
                        
                        img_name = f"{filename}_page{page_num+1}_{best_product.replace('-', '_')}_img{img_index+1}.png"
                        img_path = os.path.join("images", img_name)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        key = f"{filename}_page_{page_num+1}_{best_product}"
                        if key not in images_data:
                            images_data[key] = []
                        images_data[key].append((img_path, best_product))
                    
                    pix = None
            doc.close()
    
    return images_data

def clear_old_images():
    """Delete all existing images to force re-extraction"""
    if os.path.exists("images"):
        import shutil
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

def extract_model_names_from_text(text):
    """Extract SSC model names from user query or AI response"""
    if not text:
        return []
    
    patterns = [
        r'SSC\s*-\s*[A-Z0-9]{2,4}',
        r'SSC\s+[A-Z0-9]{2,4}',
    ]
    
    found_models = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = match.upper().replace(' ', '').replace('SSC-', 'SSC-').replace('SSC', 'SSC-')
            if not normalized.startswith('SSC-'):
                normalized = 'SSC-' + normalized.replace('SSC', '')
            found_models.add(normalized)
    
    return list(found_models)

def is_image_request(user_query):
    """Check if user is asking to see/show images"""
    image_keywords = ['show', 'image', 'picture', 'photo', 'see', 'look', 'display', 'view']
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in image_keywords)

def get_images_for_models(model_names):
    """Get images for specific model names from the images folder"""
    if not os.path.exists("images") or not model_names:
        return []
    
    found_images = []
    all_image_files = os.listdir("images")
    
    for model_name in model_names:
        model_variations = [
            model_name.replace('-', '_').replace(' ', '_'),
            model_name.replace('-', '').replace(' ', ''),
            model_name.upper(),
            model_name.replace('SSC-', '').replace('SSC_', ''),
        ]
        
        for variation in model_variations:
            for img_file in all_image_files:
                if variation.upper() in img_file.upper():
                    img_path = os.path.join("images", img_file)
                    found_images.append((img_path, model_name))
                    break
            if found_images:
                break
    
    return found_images[:6]