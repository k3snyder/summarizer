import logging
import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Dict
from .utils import pil_image_to_base64
import pandas as pd
from pdf2image import convert_from_path
import re

def extract_text_from_page(page) -> str:
    """
    Extract textual content from a pdfplumber page.
    Uses direct extraction first, and if no text is found, falls back to OCR.
    """
    try:
        text = page.extract_text()
        if text and text.strip():
            return text
        # If no text is found, use OCR on the rendered page image
        page_img = page.to_image(resolution=150)
        pil_img = page_img.original
        ocr_text = pytesseract.image_to_string(pil_img)
        return ocr_text.strip() if ocr_text else ""
    except Exception as e:
        logging.error(f"Text extraction failed: {e}")
        return ""

def text_to_table(table_text: List[List[str]]):
    """
    Convert raw table data (a list of lists) into a structured pandas DataFrame.
    Assumes the first row contains headers.
    """
    try:
        if not table_text or len(table_text) < 2:
            return None
        header = table_text[0]
        data = table_text[1:]
        df = pd.DataFrame(data, columns=header)
        return df
    except Exception as e:
        logging.error(f"Failed to convert table to dataframe: {e}")
        return None

def is_table_continued(current_table: Dict, next_table: Dict) -> bool:
    """
    Check if two tables are likely to be continuations of each other.
    
    Args:
        current_table: The table from the current page
        next_table: The table from the next page
        
    Returns:
        bool: True if tables appear to be continuous
    """
    # Debug log for table continuity check
    logging.debug(f"Checking table continuity between tables with columns: {current_table.get('columns')} and {next_table.get('columns')}")
    
    try:
        # Check if column headers match
        if current_table.get('columns') == next_table.get('columns'):
            return True
        
        # Check if the next table's first row matches the current table's columns
        # This handles cases where the header row is repeated on the next page
        if next_table.get('data') and next_table['data'][0] == current_table.get('columns'):
            return True
            
        return False
    except Exception as e:
        logging.error(f"Error checking table continuity: {e}")
        return False

def merge_continuous_tables(tables: List[Dict]) -> List[Dict]:
    """
    Merge tables that appear to be continuations of each other.
    
    Args:
        tables: List of tables from consecutive pages
        
    Returns:
        List[Dict]: Merged tables
    """
    if not tables or len(tables) < 2:
        return tables
        
    # Debug log for table merging process
    logging.debug(f"Attempting to merge {len(tables)} tables")
    
    merged_tables = []
    current_table = tables[0]
    
    for next_table in tables[1:]:
        if is_table_continued(current_table, next_table):
            # Debug log for merging tables
            logging.debug("Found continuous tables, merging...")
            
            # If next table starts with header row, skip it
            next_data = next_table['data']
            if next_data and next_data[0] == current_table['columns']:
                next_data = next_data[1:]
                
            # Merge the data
            current_table['data'].extend(next_data)
        else:
            merged_tables.append(current_table)
            current_table = next_table
    
    merged_tables.append(current_table)
    return merged_tables

def extract_tables_from_page(page, check_next_page=True) -> List[Dict]:
    """
    Extract tables from the page using multiple detection strategies.
    Filters out chart/figure artifacts that are misidentified as tables.
    """
    tables = []
    try:
        # Debug log for table extraction start
        logging.debug(f"Starting table extraction for page {page.page_number}")
        
        # Clip function to ensure bbox stays within page boundaries
        def clip_bbox(bbox):
            return (
                max(0, min(bbox[0], page.width)),
                max(0, min(bbox[1], page.height)),
                max(0, min(bbox[2], page.width)),
                max(0, min(bbox[3], page.height))
            )
        
        def is_valid_table(table_data):
            """Check if extracted table data represents a real table."""
            if not table_data or len(table_data) < 2:
                return False
            
            # Count non-empty cells
            total_cells = 0
            non_empty_cells = 0
            for row in table_data:
                for cell in row:
                    total_cells += 1
                    if cell and str(cell).strip() and cell != 'None':
                        non_empty_cells += 1
            
            # If less than 30% of cells have data, it's likely not a real table
            if total_cells == 0 or (non_empty_cells / total_cells) < 0.3:
                return False
                
            # Check if it has reasonable structure (more than 1 column with data)
            cols_with_data = 0
            for col_idx in range(len(table_data[0])):
                col_has_data = False
                for row in table_data:
                    if col_idx < len(row) and row[col_idx] and str(row[col_idx]).strip():
                        col_has_data = True
                        break
                if col_has_data:
                    cols_with_data += 1
                    
            return cols_with_data >= 2
        
        # Try multiple detection strategies
        strategies = [
            {
                "name": "lines_strict",
                "settings": {
                    "horizontal_strategy": "lines_strict",
                    "vertical_strategy": "lines_strict",
                    "intersection_y_tolerance": 3
                }
            },
            {
                "name": "lines",
                "settings": {
                    "horizontal_strategy": "lines",
                    "vertical_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1
                }
            },
            {
                "name": "text",
                "settings": {
                    "horizontal_strategy": "text",
                    "vertical_strategy": "text",
                    "snap_tolerance": 3,
                    "intersection_y_tolerance": 3
                }
            }
        ]
        
        found_valid_tables = False
        for strategy in strategies:
            if found_valid_tables:
                break
                
            logging.debug(f"Trying {strategy['name']} strategy...")
            found_tables = page.find_tables(table_settings=strategy['settings'])
            
            if not found_tables:
                continue
                
            for table in found_tables:
                # Clip the table bbox to page boundaries
                clipped_bbox = clip_bbox(table.bbox)
                if clipped_bbox[3] <= clipped_bbox[1] or clipped_bbox[2] <= clipped_bbox[0]:
                    continue
                    
                # Check if table extends to bottom of page
                table_extends_to_bottom = (clipped_bbox[3] >= page.height - 20)
                
                try:
                    # Extract the table
                    extracted = table.extract()
                    
                    if not is_valid_table(extracted):
                        logging.debug(f"Skipping invalid table from {strategy['name']} strategy")
                        continue
                        
                    # Clean the extracted data
                    cleaned_data = []
                    for row in extracted:
                        cleaned_row = []
                        for cell in row:
                            if cell is None or str(cell).strip() == '':
                                cleaned_row.append('')
                            else:
                                cleaned_row.append(str(cell).strip())
                        cleaned_data.append(cleaned_row)
                    
                    # Convert to structured format
                    if len(cleaned_data) >= 2:
                        headers = cleaned_data[0]
                        data_rows = cleaned_data[1:]
                        
                        table_dict = {
                            "columns": headers,
                            "data": data_rows,
                            "extends_to_bottom": table_extends_to_bottom
                        }
                        tables.append(table_dict)
                        found_valid_tables = True
                        logging.info(f"Successfully extracted table using {strategy['name']} strategy")
                        
                except Exception as e:
                    logging.debug(f"Error extracting table with {strategy['name']} strategy: {e}")
                    continue
        
        if not tables:
            # Try to extract chart data if no regular tables found
            chart_tables = extract_chart_data_as_tables(page)
            if chart_tables:
                tables.extend(chart_tables)
                logging.info(f"Extracted {len(chart_tables)} charts as tables")
            else:
                logging.info("No valid tables or chart data found on page")
                
    except Exception as e:
        logging.error(f"Table extraction failed: {e}")
        
    return tables

def extract_chart_data_as_tables(page) -> List[Dict]:
    """
    Extract data from charts/figures that contain percentage data.
    This handles bar charts and similar visualizations by parsing their text labels.
    """
    tables = []
    try:
        text = page.extract_text()
        if not text:
            return tables
            
        # Pattern for items with percentages
        percentage_pattern = r"([A-Za-z][A-Za-z\s\-,/]+?)\s+(\d+%)"
        
        # Split text into sections by figure titles
        figure_sections = re.split(r'FIGURE \d+\.\d+', text)
        
        for section in figure_sections:
            if not section.strip():
                continue
                
            # Look for title
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            title = lines[0].strip()
            
            # Find all percentage data in this section
            matches = re.findall(percentage_pattern, section)
            
            # Filter and clean matches
            data_items = []
            for item, percentage in matches:
                item = item.strip()
                # Filter out noise and overly short entries
                if len(item) > 5 and not any(skip in item.lower() for skip in ['share of', 'surveyed', 'source']):
                    data_items.append([item, percentage])
            
            # If we found substantial data, create a table
            if len(data_items) >= 3:
                # Remove duplicates while preserving order
                seen = set()
                unique_items = []
                for item in data_items:
                    key = (item[0], item[1])
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                
                table_dict = {
                    "columns": ["Item", "Percentage"],
                    "data": unique_items,
                    "extends_to_bottom": False,
                    "chart_data": True  # Flag to indicate this is extracted from a chart
                }
                tables.append(table_dict)
                logging.info(f"Extracted chart data with {len(unique_items)} items")
                
    except Exception as e:
        logging.error(f"Chart data extraction failed: {e}")
        
    return tables

def extract_images_from_page(page, pdf_path: str, page_number: int, dpi: int = None) -> List[str]:
    """
    Extract the entire page as a base64-encoded image using pdf2image.
    Args:
        page: The pdfplumber page object (kept for interface consistency)
        pdf_path: Path to the PDF file
        page_number: The 1-based page number to convert
        dpi: Image resolution DPI (default: uses config.image_dpi)
    Returns:
        List containing a single base64-encoded JPEG string of the full page.
    """
    from .config import config

    # Use provided DPI or fall back to config
    if dpi is None:
        dpi = config.image_dpi

    images_base64 = []
    try:
        # Convert specific page to image (first_page and last_page are 1-based)
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi  # Configurable DPI for optimal vision model performance
        )
        # Should only be one image since we're converting a single page
        if images:
            img_b64 = pil_image_to_base64(images[0])
            images_base64.append(img_b64)
    except Exception as e:
        logging.error(f"Image extraction failed for page {page_number}: {e}")
    return images_base64
