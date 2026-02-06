"""
OCR (Optical Character Recognition) Service

Service for extracting text from images and PDF files.
Supports multiple methods: PyPDF for text-based PDFs, pytesseract for images,
and GPT-4o vision for more complex bank statements.
"""

import base64
import io
import logging
from typing import Optional, Tuple

from PIL import Image
from core.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from images and PDFs"""

    def __init__(self):
        self.supported_types = settings.OCR_SUPPORTED_MIME_TYPES
        self.max_file_size = settings.OCR_MAX_FILE_SIZE

    def validate_attachment(self, mime_type: str, data: str) -> Tuple[bool, Optional[str]]:
        """
        Validate attachment before processing

        Args:
            mime_type: MIME type of the attachment
            data: Base64-encoded attachment data

        Returns:
            Tuple of (is_valid, error_message)
        """
        if mime_type not in self.supported_types:
            return False, f"Unsupported file type: {mime_type}. Supported: {self.supported_types}"

        # Decode base64 and check size
        try:
            file_data = base64.b64decode(data)
            if len(file_data) > self.max_file_size:
                max_mb = self.max_file_size / (1024 * 1024)
                return False, f"File too large. Max size: {max_mb}MB"
        except Exception as e:
            return False, f"Invalid base64 data: {e}"

        return True, None

    async def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """
        Extract text from PDF using PyPDF

        Args:
            pdf_data: PDF file bytes

        Returns:
            Extracted text
        """
        logger.info("📄 Extracting text from PDF using PyPDF")

        try:
            from pypdf import PdfReader

            pdf_file = io.BytesIO(pdf_data)
            reader = PdfReader(pdf_file)

            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n\n"
                logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1}")

            logger.info(f"✅ Extracted {len(text)} characters from {len(reader.pages)} pages")
            return text.strip()

        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF: {e}")
            raise Exception(f"PDF text extraction failed: {e}")

    async def extract_text_from_pdf_with_ocr(self, pdf_data: bytes) -> str:
        """
        Extract text from PDF by converting to images and using OCR

        Args:
            pdf_data: PDF file bytes

        Returns:
            Extracted text via OCR
        """
        logger.info("📄 Extracting text from PDF using OCR (pdf2image + tesseract)")

        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            # Convert PDF pages to images
            images = convert_from_bytes(pdf_data)
            logger.info(f"Converted PDF to {len(images)} images")

            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n\n"
                logger.debug(f"Extracted {len(page_text)} chars from page {i + 1} via OCR")

            logger.info(f"✅ Extracted {len(text)} characters via OCR from {len(images)} pages")
            return text.strip()

        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF with OCR: {e}")
            raise Exception(f"PDF OCR extraction failed: {e}")

    async def extract_text_from_image(self, image_data: bytes) -> str:
        """
        Extract text from image using pytesseract OCR

        Args:
            image_data: Image file bytes

        Returns:
            Extracted text
        """
        logger.info("🖼️ Extracting text from image using tesseract OCR")

        try:
            import pytesseract

            image = Image.open(io.BytesIO(image_data))
            logger.debug(f"Image size: {image.size}, mode: {image.mode}")

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            text = pytesseract.image_to_string(image)

            logger.info(f"✅ Extracted {len(text)} characters from image")
            return text.strip()

        except Exception as e:
            logger.error(f"❌ Failed to extract text from image: {e}")
            raise Exception(f"Image OCR extraction failed: {e}")

    async def extract_text(self, mime_type: str, data: str) -> str:
        """
        Extract text from attachment (auto-detect method)

        Args:
            mime_type: MIME type of the attachment
            data: Base64-encoded attachment data

        Returns:
            Extracted text

        Raises:
            Exception: If extraction fails or file is invalid
        """
        # Validate attachment
        is_valid, error = self.validate_attachment(mime_type, data)
        if not is_valid:
            raise Exception(error)

        # Decode base64
        file_data = base64.b64decode(data)

        # Route to appropriate extraction method
        if mime_type == "application/pdf":
            # Try PyPDF first (faster for text-based PDFs)
            try:
                text = await self.extract_text_from_pdf(file_data)
                # If we got meaningful text, return it
                if len(text.strip()) > 50:
                    return text
                # Otherwise, fall back to OCR
                logger.info("PDF text extraction yielded little text, trying OCR...")
            except Exception as e:
                logger.warning(f"PyPDF extraction failed, trying OCR: {e}")

            # Fall back to OCR
            return await self.extract_text_from_pdf_with_ocr(file_data)

        elif mime_type in ["image/png", "image/jpeg", "image/jpg"]:
            return await self.extract_text_from_image(file_data)

        else:
            raise Exception(f"Unsupported MIME type: {mime_type}")

    def prepare_image_for_vision(self, mime_type: str, data: str) -> Optional[str]:
        """
        Prepare image data for GPT-4o vision API

        Args:
            mime_type: MIME type of the attachment
            data: Base64-encoded attachment data

        Returns:
            Data URL for vision API, or None if not an image
        """
        if mime_type == "application/pdf":
            # For PDFs, we'd need to convert to image first
            # For now, we'll use OCR text extraction instead
            return None

        if mime_type in ["image/png", "image/jpeg", "image/jpg"]:
            # Vision API expects data URL format
            return f"data:{mime_type};base64,{data}"

        return None
