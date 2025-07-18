from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from scipy.stats import alpha
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
import zipfile
import tempfile

app = Flask(__name__)
CORS(app)

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Global variables for ROI selection
roi_points = []
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global roi_points, drawing
    img, window_name = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points[:] = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = img.copy()
        cv2.rectangle(temp, roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow(window_name, temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_points.append((x, y))
        temp = img.copy()
        cv2.rectangle(temp, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.imshow(window_name, temp)

def select_roi_on_image(image, window_name="Select ROI"):
    global roi_points
    roi_points = []

    max_dim = 800  # Smaller for web interface
    scale = 1.0
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    clone = image.copy()
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, draw_rectangle, param=(clone, window_name))
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if len(roi_points) != 2:
        # If no ROI selected, use full image
        return 0, 0, image.shape[1], image.shape[0]

    (x1, y1), (x2, y2) = roi_points
    x, y = int(min(x1, x2) / scale), int(min(y1, y2) / scale)
    w, h = int(abs(x2 - x1) / scale), int(abs(y2 - y1) / scale)
    return x, y, w, h

def align_with_orb(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is None or des2 is None:
        raise ValueError("Not enough features detected.")

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        raise ValueError("Not enough matches for homography.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    height, width = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, H, (width, height))
    return aligned, H

def extract_patch(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y + h, x:x + w], (x, y, w, h)

def remove_text_with_paddleocr(image, tile_size=1000, overlap=100):
    result_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]
    text_boxes = []

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)
            tile = image[y:end_y, x:end_x]

            try:
                # result = ocr.ocr(tile, cls=True)
                result = ocr.predict(tile)
                
                if result is not None and len(result) > 0:
                    for idx in range(len(result)):
                        if result[idx] is None:
                            continue
                        for line in result[idx]:
                            try:
                                if len(line) >= 1:
                                    box = line[0]
                                    if isinstance(box, list) and len(box) == 4:
                                        adjusted_box = [[point[0] + x, point[1] + y] for point in box]
                                        text_boxes.append((adjusted_box, line[1]))
                                        pts = np.array(adjusted_box).astype(np.int32).reshape((-1, 1, 2))
                                        cv2.fillPoly(mask, [pts], 255)
                            except Exception as e:
                                print(f"Error processing detection: {e}")
            except Exception as e:
                print(f"Error in OCR processing: {e}")

    if np.max(mask) > 0:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    inpainted = cv2.inpaint(result_image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted, mask, text_boxes

def is_blank(patch, threshold=10):
    return np.std(patch) < threshold

def is_modified(patch1, patch2, threshold=0.6):
    if patch1.shape[:2] != patch2.shape[:2]:
        h, w = min(patch1.shape[0], patch2.shape[0]), min(patch1.shape[1], patch2.shape[1])
        patch1, patch2 = patch1[:h, :w], patch2[:h, :w]

    gray1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    min_size = min(gray1.shape[0], gray1.shape[1], gray2.shape[0], gray2.shape[1])

    if min_size < 7:
        if min_size < 3:
            mse = np.mean((gray1 - gray2) ** 2)
            max_mse = 255 ** 2
            similarity = 1 - (mse / max_mse)
        else:
            win_size = min_size if min_size % 2 == 1 else min_size - 1
            score, _ = ssim(gray1, gray2, full=True, win_size=win_size)
            similarity = score
    else:
        score, _ = ssim(gray1, gray2, full=True)
        similarity = score

    return similarity < threshold

def classify_difference(patch1, patch2):
    if is_blank(patch2, threshold=10):
        return "missing"
    elif is_modified(patch1, patch2, threshold=0.6):
        return "modified"
    else:
        return "unchanged"

def compare_regions(region1, region2):
    region2_aligned, H = align_with_orb(region1, region2)
    gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(region2_aligned, cv2.COLOR_BGR2GRAY)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_h1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detect_h2 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_v1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    detect_v2 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    lines1 = cv2.add(detect_h1, detect_v1)
    lines2 = cv2.add(detect_h2, detect_v2)

    line_diff = cv2.absdiff(lines1, lines2)
    normal_diff = cv2.absdiff(gray1, gray2)
    diff = cv2.addWeighted(normal_diff, 0.5, line_diff, 1.0, 0)

    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    blurred_diff = cv2.GaussianBlur(norm_diff, (5, 5), 0)

    thresh_diff = cv2.adaptiveThreshold(
        blurred_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)

    _, global_mask = cv2.threshold(norm_diff, 25, 255, cv2.THRESH_BINARY)
    thresh_diff = cv2.bitwise_and(thresh_diff, global_mask)

    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh_diff, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_cleaned = np.zeros_like(dilated)

    for cnt in contours:
        cv2.drawContours(mask_cleaned, [cnt], -1, 255, -1)

    color_mask = np.zeros_like(region2_aligned)
    color_mask[mask_cleaned > 0] = (0, 180, 180)
    highlighted = cv2.addWeighted(region2_aligned, 0.7, color_mask, 0.3, 0)

    comparison = np.hstack([region1, highlighted])
    return region1, highlighted, comparison, mask_cleaned, region2_aligned, H

def analyze_differences(region1, mask_cleaned, region2_aligned, highlighted):
    results = []
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = highlighted.copy()
    annotated_side_by_side = np.hstack((region1.copy(), highlighted.copy()))

    for i, cnt in enumerate(contours):
        patch1, (x, y, w, h) = extract_patch(region1, cnt)
        patch2 = region2_aligned[y:y + h, x:x + w]

        classification = classify_difference(patch1, patch2)
        if classification == "unchanged":
            continue

        color_map = {
            "missing": (0, 0, 255),
            "modified": (0, 255, 0),
            "unchanged": (0, 180, 180)
        }
        color = color_map.get(classification, (0, 180, 180))

        contour_mask = np.zeros_like(overlay)
        cv2.drawContours(contour_mask, [cnt], -1, color, thickness=cv2.FILLED)

        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1.0, contour_mask, alpha, 0)

        offset = region1.shape[1]
        right_contour_mask = np.zeros_like(region1)
        cv2.drawContours(right_contour_mask, [cnt], -1, color, thickness=cv2.FILLED)

        mask_region = annotated_side_by_side[:, offset:]
        blended_region = cv2.addWeighted(mask_region, 1.0, right_contour_mask, alpha, 0)
        annotated_side_by_side[:, offset:] = blended_region

        results.append({
            "box": (x, y, w, h),
            "classification": classification
        })

    annotated_side_by_side = draw_legend(annotated_side_by_side)
    return results, overlay, annotated_side_by_side

def draw_legend(image):
    legend_items = [
        ("Missing", (0, 0, 255)),
        ("Modified", (0, 255, 0)),
        ("Other", (0, 255, 255))
    ]

    start_x, start_y = 10, image.shape[0] - 100
    box_size = 20
    spacing = 35

    for i, (label, color) in enumerate(legend_items):
        y = start_y + i * spacing
        cv2.rectangle(image, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        cv2.putText(image, label, (start_x + box_size + 10, y + box_size - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.route('/compare-images', methods=['POST'])
def compare_images():
    try:
        # Get uploaded files
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both images are required'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        print("Image 1 uploaded successfully:", file1.filename)
        print("Image 2 uploaded successfully:", file2.filename)

        # Convert files to OpenCV images
        img1_bytes = np.frombuffer(file1.read(), np.uint8)
        img2_bytes = np.frombuffer(file2.read(), np.uint8)
        
        image1 = cv2.imdecode(img1_bytes, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(img2_bytes, cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            return jsonify({'error': 'Could not decode images'}), 400
        
        print("Images decoded successfully.")

        # For web interface, we'll use the full images as ROI
        # In a more advanced version, you could implement ROI selection via coordinates
        # region1_original = image1.copy()
        # region2_original = image2.copy()

        # image1 = cv2.imread(image1)
        # image2 = cv2.imread(image2)

        if image1 is None or image2 is None:
            print("Error loading images.")
            return

        # Select ROI on the first image
        x1, y1, w1, h1 = select_roi_on_image(image1)
        x2, y2, w2, h2 = select_roi_on_image(image2)

        # Extract the selected region from both images
        region1_original = image1[y1:y1 + h1, x1:x1 + w1].copy()
        region2_original = image2[y2:y2 + h2, x2:x2 + w2].copy()

        # Remove text from both images
        print("Detecting text in first image...")
        region1_no_text, text_mask1, text_boxes1 = remove_text_with_paddleocr(region1_original)
        
        print("Detecting text in second image...")
        region2_no_text, text_mask2, text_boxes2 = remove_text_with_paddleocr(region2_original)

        # Compare regions
        region1_clean, highlighted_clean, comparison_clean, mask_cleaned, region2_aligned_clean, H = compare_regions(
            region1_no_text, region2_no_text)

        # Align original second image
        region2_original_aligned = cv2.warpPerspective(
            region2_original, H, (region1_original.shape[1], region1_original.shape[0]))

        # Analyze differences
        print("Analyzing differences...")
        results, overlay_clean, annotated_comparison_clean = analyze_differences(
            region1_clean, mask_cleaned, region2_aligned_clean, highlighted_clean)

        # testing
        print("Results:", results)

        # Create final comparison with highlights
        drawing_diff_mask = np.zeros(region1_original.shape[:2], dtype=np.uint8)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            classification = classify_difference(
                extract_patch(region1_clean, cnt)[0],
                extract_patch(region2_aligned_clean, cnt)[0])

            if classification != "unchanged":
                cv2.drawContours(drawing_diff_mask, [cnt], -1, 255, -1)

        # Create color mask and final comparison
        color_mask = np.zeros_like(region1_original)
        color_mask[drawing_diff_mask > 0] = (0, 255, 255)
        highlighted_with_text = cv2.addWeighted(region2_original_aligned, 0.7, color_mask, 0.3, 0)
        final_comparison = np.hstack([region1_original, highlighted_with_text])
        final_comparison = draw_legend(final_comparison)

        print("Saving results...")

        # Convert images to base64 for web display
        response_data = {
            'success': True,
            'results': results,
            'images': {
                'original1': image_to_base64(region1_original),
                'original2': image_to_base64(region2_original_aligned),
                'highlighted': image_to_base64(highlighted_with_text),
                'comparison': image_to_base64(final_comparison)
            },
            'summary': {
                'total_differences': len(results),
                'missing_count': sum(1 for r in results if r['classification'] == 'missing'),
                'modified_count': sum(1 for r in results if r['classification'] == 'modified')
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
