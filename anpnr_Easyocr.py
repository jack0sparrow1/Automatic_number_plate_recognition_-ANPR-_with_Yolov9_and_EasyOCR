# Save this as anpr_reader.py (NOT easyocr.py)
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_license_plate(image_path):
    """
    Specifically optimized for license plate reading
    """
    # Initialize EasyOCR reader (only English is usually sufficient for license plates)
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Read text from image
    # For license plates, we want high accuracy, so we'll use more strict parameters
    results = reader.readtext(
        img_rgb,
        paragraph=False,  # Don't combine text
        text_threshold=0.5,  # Lower threshold to catch more text
        low_text=0.3,  # Lower threshold for text detection
        link_threshold=0.3,  # Lower threshold for linking characters
        canvas_size=2560,  # Maximum image size
        mag_ratio=1.5,  # Magnification ratio
        slope_ths=0.1,  # Maximum slope for text boxes
        ycenter_ths=0.5,  # Vertical center threshold
        height_ths=0.5,  # Height threshold
        width_ths=0.8,  # Width threshold for license plates (wider characters)
        add_margin=0.1,  # Additional margin
        decoder='greedy',  # Use greedy decoder for speed
        beamWidth=5,  # Beam width
        batch_size=1,
    )
    
    return results, img_rgb

def visualize_results(results, img):
    """
    Visualize the detected text on the image
    """
    # Create a copy of the image
    img_with_boxes = img.copy()
    
    print("\n" + "="*50)
    print("DETECTED TEXT:")
    print("="*50)
    
    for detection in results:
        bbox, text, confidence = detection
        
        # Convert bbox points to integers
        bbox = np.array(bbox, dtype=int)
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, 
                     tuple(bbox[0]), 
                     tuple(bbox[2]), 
                     (0, 255, 0), 2)
        
        # Put text above bounding box
        cv2.putText(img_with_boxes, 
                   f"{text} ({confidence:.2f})", 
                   (bbox[0][0], bbox[0][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (255, 0, 0), 
                   2)
        
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Position: {bbox}")
        print("-" * 30)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title('License Plate Detection Results')
    plt.show()
    
    return img_with_boxes

def extract_license_plate_number(results):
    """
    Extract and clean the most likely license plate number
    """
    if not results:
        return None
    
    # Filter results by confidence and clean the text
    valid_plates = []
    
    for detection in results:
        bbox, text, confidence = detection
        
        # Clean the text: remove spaces and special characters, convert to uppercase
        cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
        
        # License plates are typically 5-10 characters
        if 4 <= len(cleaned_text) <= 10 and confidence > 0.3:
            valid_plates.append((cleaned_text, confidence, bbox))
    
    # Sort by confidence (highest first)
    valid_plates.sort(key=lambda x: x[1], reverse=True)
    
    return valid_plates

def main():
    # Path to your image
    image_path = r'C:\\MLprojects\Automatic_number_plate_recognition_(ANPR)_with_Yolov9_and_EasyOCR\\image.webp'
    
    print("Reading license plate from image...")
    
    # Read the license plate
    results, img = read_license_plate(image_path)
    
    if results:
        # Visualize results
        visualize_results(results, img)
        
        # Extract license plate number
        plates = extract_license_plate_number(results)
        
        print("\n" + "="*50)
        print("LICENSE PLATE RESULTS:")
        print("="*50)
        
        if plates:
            for i, (plate, confidence, bbox) in enumerate(plates):
                print(f"{i+1}. License Plate: {plate}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Position: {bbox}")
                print()
            
            print(f"Most likely license plate: {plates[0][0]}")
        else:
            print("No valid license plate detected")
    else:
        print("No text detected in the image")

if __name__ == "__main__":
    main()