import asyncio
import os
import logging
import json
import pdfplumber
from .models import PDFDocument, ParserOutput
from .parsers import extract_text_from_page, extract_tables_from_page, extract_images_from_page, merge_continuous_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_single_page(pdf, page_index: int, doc_title: str, pdf_path: str, skip_tables: bool = False, skip_images: bool = False, image_dpi: int = None) -> ParserOutput:
    """
    Process a single PDF page by extracting text, tables, and images.
    Args:
        pdf: The pdfplumber PDF object
        page_index: The 1-based page index
        doc_title: The title of the PDF
        pdf_path: Path to the PDF file for image extraction
        skip_tables: If True, skip table extraction
        skip_images: If True, skip image extraction
        image_dpi: Image resolution DPI (default: uses config)
    """
    try:
        page = pdf.pages[page_index - 1]  # 0-based index
        text = extract_text_from_page(page)

        tables = []
        if not skip_tables:
            tables = extract_tables_from_page(page)

        images = []
        if not skip_images:
            images = extract_images_from_page(page, pdf_path, page_index, dpi=image_dpi)
            
        return ParserOutput(
            chunk_id=f"chunk_{page_index}",
            doc_title=doc_title,
            text=text,
            tables=tables,
            image_base64=images
        )
    except Exception as e:
        logging.error(f"Error processing page {page_index}: {e}")
        return ParserOutput(
            chunk_id=f"chunk_{page_index}",
            doc_title=doc_title,
            text="",
            tables=[],
            image_base64=[]
        )

async def process_pdf(filepath: str, skip_tables: bool = False, skip_images: bool = False, image_dpi: int = None):
    """
    Process the entire PDF by extracting content from each page and generating a unified JSON output.
    Handles tables that span across multiple pages.

    Args:
        filepath: Path to PDF file
        skip_tables: Skip table extraction
        skip_images: Skip image extraction
        image_dpi: Image resolution DPI (72=low, 144=medium, 200+=high quality)
    """
    if not os.path.exists(filepath):
        logging.error(f"File {filepath} does not exist.")
        return

    doc_title = os.path.basename(filepath)
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            pages_output = []
            pending_tables = []  # Store tables that might continue to next page
            
            # First pass: extract all content from pages
            for i in range(1, total_pages + 1):
                # Debug log for page processing
                logging.debug(f"Processing page {i} of {total_pages}")
                
                output = process_single_page(pdf, i, doc_title, filepath, skip_tables, skip_images, image_dpi)
                
                # Handle table continuity only if we extracted tables
                if not skip_tables and pending_tables and output.tables:
                    # Try to merge with pending tables from previous page
                    merged_tables = merge_continuous_tables(pending_tables + output.tables)
                    
                    # Update the previous page's tables if merging occurred
                    if len(merged_tables) < len(pending_tables) + len(output.tables):
                        pages_output[-1].tables = merged_tables[:-1]  # All but last table
                        output.tables = merged_tables[-1:]  # Last table only
                
                # Check if any tables extend to bottom of page
                if not skip_tables:
                    pending_tables = [t for t in output.tables if t.get('extends_to_bottom', False)]
                    if pending_tables:
                        # Remove the extends_to_bottom flag as it's no longer needed
                        for t in pending_tables:
                            t.pop('extends_to_bottom', None)
                else:
                    pending_tables = []
                
                pages_output.append(output)
            
            # Create document metadata
            pdf_doc = PDFDocument(
                document_id=f"doc_{doc_title}",
                filename=doc_title,
                total_pages=total_pages,
                metadata={}  # Additional metadata can be added here if needed.
            )
            
            # Combine the document metadata and per-page outputs
            result = {
                "document": pdf_doc.model_dump(),
                "pages": [page.model_dump() for page in pages_output]
            }
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            
            # Use fixed output filename
            output_path = os.path.join('output', 'output_parsed.json')
            
            # Save the JSON output to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logging.info(f"PDF parsing complete. Output saved to: {output_path}")
            
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF Parser Flow Pipeline")
    parser.add_argument("filepath", help="Path to the PDF file")
    parser.add_argument("--skip-tables", action="store_true", help="Skip table extraction")
    parser.add_argument("--skip-images", action="store_true", help="Skip image extraction")
    args = parser.parse_args()
    asyncio.run(process_pdf(args.filepath, skip_tables=args.skip_tables, skip_images=args.skip_images))

if __name__ == "__main__":
    main()
