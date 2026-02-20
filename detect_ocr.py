"""
High Accuracy License Plate Detection and OCR with YOLOv9 + EasyOCR
Optimized for Indian License Plates
"""

import argparse
import sys
import re
from pathlib import Path

# Add yolov9 to path
YOLO_PATH = Path(__file__).parent / 'yolov9'
if not YOLO_PATH.exists():
    print("ERROR: yolov9 folder not found!")
    sys.exit(1)

sys.path.insert(0, str(YOLO_PATH))

import cv2
import torch

# Install EasyOCR if not installed
try:
    import easyocr
except ImportError:
    print("Installing EasyOCR...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'easyocr'])
    import easyocr

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


# Common OCR character substitutions for Indian plates
OCR_CORRECTIONS = {
    # Digits misread as letters
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    # Letters misread as digits
    'O': '0',
    'I': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'L': '1',
}

def correct_ocr_errors(text):
    """Correct common OCR mistakes using context"""
    corrected = ""
    for i, char in enumerate(text):
        # Keep character as is for now
        corrected += char
    return corrected

def correct_indian_plate(text):
    text = text.replace(" ", "").upper()
    
    pattern = r'^([A-Z]{2})(\d{2})([A-Z]{0,2})(\d{4})$'
    match = re.match(pattern, text)
    
    if match:
        state = match.group(1)
        district = match.group(2)
        series = match.group(3)
        number = match.group(4)
        
        if series:
            return f"{state} {district} {series} {number}"
        else:
            return f"{state} {district} {number}"
    
    return text


class LicensePlateDetector:

    def __init__(self, weights, conf_thres=0.25, iou_thres=0.45, device='', debug=False):

        print(f"Loading model: {weights}")

        self.device = select_device(device)

        self.model = DetectMultiBackend(
            weights,
            device=self.device,
            dnn=False,
            fp16=False
        )

        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

        self.imgsz = check_img_size(640, s=self.stride)

        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug = debug

        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(
            ['en'],
            gpu=torch.cuda.is_available()
        )

        print("✅ Ready!\n")


    def preprocess_plate(self, plate):

        # Convert to grayscale
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Upscale image (VERY IMPORTANT)
        scale = 3
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Noise removal
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Increase contrast
        gray = cv2.equalizeHist(gray)

        # Threshold
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Convert to RGB
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        return processed

    def read_plate_section(self, plate_image, section_type='full', confidence_threshold=0.3):
        """
        Read specific section of the plate with optimized OCR settings
        section_type: 'full', 'state' (letters), 'district' (digits), 'number' (digits), 'series' (letters)
        """
        
        allowlist_map = {
            'state': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'district': '0123456789',
            'series': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'number': '0123456789',
            'full': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        }
        
        allowlist = allowlist_map.get(section_type, allowlist_map['full'])
        
        results = self.reader.readtext(
            plate_image,
            allowlist=allowlist,
            decoder='beamsearch',
            batch_size=1,
            detail=1,  # Get detail for confidence
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7
        )
        
        # Filter by confidence
        high_conf_results = []
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                high_conf_results.append(text.upper())
        
        return ''.join(high_conf_results), results
    
    def smart_ocr_plate(self, plate_image):
        """
        Perform OCR on the entire plate and apply corrections based on position
        """
        processed_plate = self.preprocess_plate(plate_image)
        
        # Try to read with lower confidence threshold first
        text, raw_results = self.read_plate_section(processed_plate, 'full', confidence_threshold=0.15)
        
        print(f"[DEBUG] Raw OCR text: '{text}'")
        print(f"[DEBUG] Raw results count: {len(raw_results)}")
        
        if raw_results:
            for i, (bbox, char_text, conf) in enumerate(raw_results):
                print(f"  [{i}] '{char_text}' (confidence: {conf:.2f})")
        
        if not text:
            print("[DEBUG] No text detected by OCR")
            return None
        
        # Try to parse and correct based on expected positions
        text_clean = text.replace(" ", "")
        print(f"[DEBUG] Cleaned text: '{text_clean}' (length: {len(text_clean)})")
        
        if len(text_clean) < 6:
            print(f"[DEBUG] Text too short (need at least 6 chars)")
            return text_clean  # Return whatever we have
        
        # First 2 chars should be letters (state code)
        state = ""
        for c in text_clean[:2]:
            if c.isalpha():
                state += c
        
        if not state:
            state = text_clean[:2]
        
        state = state.upper()[:2]
        print(f"[DEBUG] State: '{state}'")
        
        # Next 2 chars should be digits (district code)
        remaining = text_clean[2:]
        district = ""
        
        for c in remaining[:4]:  # Look in next 4 chars for 2 digits
            if c.isdigit():
                district += c
                if len(district) == 2:
                    break
            elif len(district) < 2:
                # Try to convert letter to digit
                converted = self._digit_from_letter(c)
                if converted:
                    district += converted
                    if len(district) == 2:
                        break
        
        print(f"[DEBUG] District: '{district}'")
        
        if len(district) < 2:
            print(f"[DEBUG] Could not find 2 district digits")
        
        # Find position after district in remaining
        chars_consumed = 0
        for i, c in enumerate(remaining):
            if c.isdigit() or (c.isalpha() and self._digit_from_letter(c)):
                chars_consumed += 1
                if chars_consumed == len(district):
                    remaining = remaining[i+1:]
                    break
        
        # Try to identify series and number from what's left
        series = ""
        number = ""
        
        # Get letters first (potential series, max 2)
        for c in remaining:
            if c.isalpha() and len(series) < 2:
                series += c
            elif c.isdigit() or (len(series) > 0 and self._digit_from_letter(c)):
                break
        
        # Rest should be numbers (4 digits)
        remaining_for_number = remaining[len(series):]
        for c in remaining_for_number:
            if len(number) < 4:
                if c.isdigit():
                    number += c
                else:
                    converted = self._digit_from_letter(c)
                    if converted:
                        number += converted
        
        print(f"[DEBUG] Series: '{series}', Number: '{number}'")
        
        # Build result based on what we have
        if len(state) == 2:
            if len(district) == 2 and len(number) == 4:
                if len(series) == 2:
                    result = f"{state} {district} {series} {number}"
                else:
                    result = f"{state} {district} {number}"
                print(f"[DEBUG] Successfully parsed: {result}")
                return result
            else:
                # Return what we have with spaces
                print(f"[DEBUG] Partial match, returning: {state} {district} {series} {number}")
                return f"{state} {district} {series} {number}".strip()
        
        # Fallback: return cleaned text
        print(f"[DEBUG] Parsing failed, returning raw text")
        return text_clean
    
    def _digit_from_letter(self, char):
        """Try to convert a misread letter to a digit"""
        mapping = {
            'O': '0',
            'o': '0',
            'I': '1',
            'i': '1',
            'Z': '2',
            'z': '2',
            'S': '5',
            's': '5',
            'B': '8',
            'b': '8',
            'G': '6',
            'g': '6',
            'L': '1',
            'l': '1',
        }
        return mapping.get(char, "")


    def detect(self, image_path):

        dataset = LoadImages(
            image_path,
            img_size=self.imgsz,
            stride=self.stride,
            auto=self.pt
        )

        for path, im, im0s, vid_cap, s in dataset:

            im = torch.from_numpy(im).to(self.device)
            im = im.float() / 255.0

            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im, augment=False, visualize=False)

            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]

            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                max_det=300
            )

            im0 = im0s.copy()

            for det in pred:

                if len(det):

                    det[:, :4] = scale_boxes(
                        im.shape[2:],
                        det[:, :4],
                        im0.shape
                    ).round()

                    for *xyxy, conf, cls in reversed(det):

                        x1, y1, x2, y2 = map(int, xyxy)

                        # Add padding
                        padding = 10

                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(im0.shape[1], x2 + padding)
                        y2 = min(im0.shape[0], y2 + padding)

                        plate = im0[y1:y2, x1:x2]

                        if plate.size == 0:
                            continue

                        # Save preprocessed plate for debugging
                        if self.debug:
                            import os
                            os.makedirs('debug', exist_ok=True)
                            processed_plate = self.preprocess_plate(plate)
                            cv2.imwrite(f'debug/plate_{x1}_{y1}.jpg', processed_plate)

                        # Smart OCR with corrections
                        text = self.smart_ocr_plate(plate)

                        if text:
                            print(f"Detected Plate: {text}")
                        else:
                            print("No valid plate detected")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--weights',
        type=str,
        default='best.pt'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45
    )

    parser.add_argument(
        '--device',
        type=str,
        default=''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save preprocessed plates for debugging'
    )

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"❌ Weights not found: {args.weights}")
        return

    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return

    detector = LicensePlateDetector(
        args.weights,
        args.conf,
        args.iou,
        args.device,
        args.debug
    )

    detector.detect(args.image)


if __name__ == "__main__":
    main()