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
    # Letters to Digits (for district and number positions)
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'L': '1', 'A': '4', 'T': '7', 'H': '4', 'K': '4',
    # Digits to Letters (for state and series positions)
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '4': 'A', '6': 'G', '7': 'T',
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
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Subtle blur to join broken parts
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        return gray

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
        Perform OCR using two passes:
        1. A full-plate pass to get rough text and structure
        2. Per-section passes with allowlists for better accuracy
        """
        processed_plate = self.preprocess_plate(plate_image)
        h, w = processed_plate.shape[:2]

        # --- Pass 1: Full plate to get rough structure ---
        results = self.reader.readtext(
            processed_plate,
            paragraph=False,
            decoder='beamsearch',
            contrast_ths=0.1,
            adjust_contrast=1.2,
            text_threshold=0.3
        )
        raw_text = "".join([res[1] for res in results]).upper()
        clean_text = "".join([c for c in raw_text if c.isalnum()])
        print(f"[DEBUG] Raw OCR (Pass 1): '{clean_text}'")

        # Strip known plate suffixes (IND, BH, etc.) that appear on HSRP/smart plates
        KNOWN_SUFFIXES = ('IND', 'BH')
        for suffix in KNOWN_SUFFIXES:
            if clean_text.endswith(suffix):
                clean_text = clean_text[:-len(suffix)]
                print(f"[DEBUG] Stripped suffix '{suffix}': '{clean_text}'")
                break

        # --- Pass 2: Per-region reads ---
        # Split the plate into:
        #  Left half  = State + District (e.g. MH12)   → letters + digits
        #  Right half = Series + Number  (e.g. DE1433)  → letters + digits
        # Then further:
        #  Right-right quarter = Number only (e.g. 1433) → digits only
        left_half  = processed_plate[:, :w//2]
        right_half = processed_plate[:, w//2:]
        right_right = processed_plate[:, int(w * 0.6):]   # last ~40% for number

        def read_region(region, allowlist, label):
            res = self.reader.readtext(
                region,
                allowlist=allowlist,
                decoder='beamsearch',
                paragraph=False,
                contrast_ths=0.1,
                adjust_contrast=1.2,
                text_threshold=0.2
            )
            text = "".join([r[1] for r in res]).upper()
            text = "".join([c for c in text if c.isalnum()])
            print(f"[DEBUG] {label}: '{text}'")
            return text

        left_text  = read_region(left_half,  'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 'Left (State+Dist)')
        number_text = read_region(right_right, '0123456789', 'Number (digits only)')
        # --- Use Pass 1 clean_text as primary, but substitute number from Pass 2 ---
        # Determine number from targetted pass (digits only is more reliable)
        if len(number_text) >= 4:
            number_raw = number_text[-4:]
        elif len(clean_text) >= 4:
            number_raw = clean_text[-4:]
        else:
            number_raw = clean_text[-4:] if len(clean_text) >= 4 else clean_text

        # Apply digit-only corrections to number
        number = ""
        for c in number_raw:
            if c.isalpha():
                if c in ('A', 'I', 'L', 'Z'): number += '1' if c in ('A', 'I', 'L') else '2'
                else: number += self._digit_from_letter(c)
            else:
                number += c

        # --- Build state, district, series from clean_text ---
        # Anchor: use number to determine where number starts in clean_text
        n = len(clean_text)

        # State (pos 0-1): always letters
        state_raw = clean_text[:2]
        state = ""
        for i, c in enumerate(state_raw):
            if c.isdigit():
                state += self._letter_from_digit(c)
            elif c in ('K', 'I', '1') and i == 0:
                state += 'M'
            else:
                state += c

        # Middle = everything after state and before the 4-digit number
        # We know the number from the digits pass, so middle = clean_text[2 : n-4]
        middle = clean_text[2:-4] if n > 6 else clean_text[2:]

        # District: first 2 chars of middle
        district = ""
        series   = ""

        if len(middle) >= 2:
            district_raw = middle[:2]
            series_raw   = middle[2:]

            # District digits — some states (e.g. DL) use alphanumeric like "3C"
            for c in district_raw:
                if c.isalpha():
                    mapped = self._digit_from_letter(c)
                    # If no mapping exists, keep the letter (valid alphanumeric district)
                    district += mapped
                else:
                    district += c

            # Series: should be letters
            for c in series_raw:
                if c.isdigit():
                    series += self._letter_from_digit(c)
                else:
                    series += c
        else:
            district = middle

        result = f"{state} {district} {series} {number}".strip()
        result = " ".join(result.split())
        print(f"[DEBUG] Final Parsed: {result}")
        return result
    
    def _letter_from_digit(self, char):
        return OCR_CORRECTIONS.get(char, char)
    
    def _digit_from_letter(self, char):
        return OCR_CORRECTIONS.get(char, char)


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