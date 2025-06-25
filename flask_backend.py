from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
from scipy.stats import alpha
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
import tempfile
import shutil
from datetime import datetime, timedelta
import threading
import time
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
CLEANUP_INTERVAL = 3600  # 1 hour
FILE_RETENTION = 7200  # 2 hours

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Global storage for temporary files and their metadata
temp_files = {}

class ImageComparisonEngine:
    def __init__(self):
        self.roi_points = []
        self.drawing = False

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def align_with_orb(self, img1, img2):
        """Align two images using ORB feature matching"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(10000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            raise ValueError("Not enough features detected for alignment.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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

    def extract_patch(self, image, contour):
        """Extract a patch from image based on contour"""
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y + h, x:x + w], (x, y, w, h)

    def remove_text_with_paddleocr(self, image, tile_size=1000, overlap=100):
        """Remove text from image using PaddleOCR"""
        result_image = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        height, width = image.shape[:2]
        text_boxes = []

        # Process the image in tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                end_y = min(y + tile_size, height)
                end_x = min(x + tile_size, width)
                tile = image[y:end_y, x:end_x]

                try:
                    result = ocr.ocr(tile, cls=True)
                    if result is not None and len(result) > 0:
                        for idx in range(len(result)):
                            if result[idx] is None:
                                continue
                            for line in result[idx]:
                                if len(line) >= 1:
                                    box = line[0]
                                    if isinstance(box, list) and len(box) == 4:
                                        adjusted_box = [[point[0] + x, point[1] + y] for point in box]
                                        text_boxes.append((adjusted_box, line[1]))
                                        pts = np.array(adjusted_box).astype(np.int32).reshape((-1, 1, 2))
                                        cv2.fillPoly(mask, [pts], 255)
                except Exception as e:
                    print(f"Error processing OCR tile: {e}")

        # Apply inpainting
        if np.max(mask) > 0:
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            result_image = cv2.inpaint(result_image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        return result_image, mask, text_boxes

    def is_blank(self, patch, threshold=10):
        """Check if a patch is blank/empty"""
        return np.std(patch) < threshold

    def is_modified(self, patch1, patch2, threshold=0.6):
        """Check if patches are significantly different"""
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

    def classify_difference(self, patch1, patch2):
        """Classify the type of difference between patches"""
        if self.is_blank(patch2, threshold=10):
            return "missing"
        elif self.is_modified(patch1, patch2, threshold=0.6):
            return "modified"
        else:
            return "unchanged"

    def compare_regions(self, region1, region2):
        """Compare two image regions and highlight differences"""
        try:
            region2_aligned, H = self.align_with_orb(region1, region2)
        except Exception as e:
            print(f"Alignment failed, using original images: {e}")
            region2_aligned = cv2.resize(region2, (region1.shape[1], region1.shape[0]))
            H = np.eye(3)

        # Convert to grayscale
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(region2_aligned, cv2.COLOR_BGR2GRAY)

        # Detect lines
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

    def analyze_differences(self, region1, mask_cleaned, region2_aligned, highlighted):
        """Analyze and classify differences"""
        results = []
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = highlighted.copy()
        annotated_side_by_side = np.hstack((region1.copy(), highlighted.copy()))

        for i, cnt in enumerate(contours):
            patch1, (x, y, w, h) = self.extract_patch(region1, cnt)
            patch2 = region2_aligned[y:y + h, x:x + w]

            classification = self.classify_difference(patch1, patch2)
            if classification == "unchanged":
                continue

            color_map = {
                "missing": (0, 0, 255),    # Red
                "modified": (0, 255, 0),   # Green
                "unchanged": (0, 180, 180) # Yellow
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

        annotated_side_by_side = self.draw_legend(annotated_side_by_side)
        return results, overlay, annotated_side_by_side

    def draw_legend(self, image):
        """Draw legend on the comparison image"""
        legend_items = [
            ("Missing", (0, 0, 255)),   # Red
            ("Modified", (0, 255, 0)),  # Green
            ("Other", (0, 255, 255))    # Yellow
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

    def process_comparison(self, image1_path, image2_path, roi1=None, roi2=None):
        """Main processing function"""
        try:
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            if image1 is None or image2 is None:
                raise ValueError("Error loading images")

            # If ROI not specified, use entire image
            if roi1 is None:
                roi1 = (0, 0, image1.shape[1], image1.shape[0])
            if roi2 is None:
                roi2 = (0, 0, image2.shape[1], image2.shape[0])

            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2

            region1_original = image1[y1:y1 + h1, x1:x1 + w1].copy()
            region2_original = image2[y2:y2 + h2, x2:x2 + w2].copy()

            # Remove text
            print("Processing text removal...")
            region1_no_text, text_mask1, text_boxes1 = self.remove_text_with_paddleocr(region1_original)
            region2_no_text, text_mask2, text_boxes2 = self.remove_text_with_paddleocr(region2_original)

            # Compare regions
            print("Comparing regions...")
            region1_clean, highlighted_clean, comparison_clean, mask_cleaned, region2_aligned_clean, H = \
                self.compare_regions(region1_no_text, region2_no_text)

            # Align original images
            region2_original_aligned = cv2.warpPerspective(
                region2_original, H, (region1_original.shape[1], region1_original.shape[0]))

            # Analyze differences
            print("Analyzing differences...")
            results, overlay_clean, annotated_comparison_clean = \
                self.analyze_differences(region1_clean, mask_cleaned, region2_aligned_clean, highlighted_clean)

            # Create final outputs
            drawing_diff_mask = np.zeros(region1_original.shape[:2], dtype=np.uint8)
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                classification = self.classify_difference(
                    self.extract_patch(region1_clean, cnt)[0],
                    self.extract_patch(region2_aligned_clean, cnt)[0])

                if classification != "unchanged":
                    cv2.drawContours(drawing_diff_mask, [cnt], -1, 255, -1)

            color_mask = np.zeros_like(region1_original)
            color_mask[drawing_diff_mask > 0] = (0, 255, 255)

            highlighted_with_text = cv2.addWeighted(region2_original_aligned, 0.7, color_mask, 0.3, 0)
            final_comparison = np.hstack([region1_original, highlighted_with_text])
            final_comparison = self.draw_legend(final_comparison)

            return {
                'final_comparison': final_comparison,
                'highlighted_with_text': highlighted_with_text,
                'region1_original': region1_original,
                'results': results,
                'text_boxes1': text_boxes1,
                'text_boxes2': text_boxes2
            }

        except Exception as e:
            raise Exception(f"Error in image processing: {str(e)}")

# Initialize the comparison engine
comparison_engine = ImageComparisonEngine()

def cleanup_old_files():
    """Clean up old temporary files"""
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getmtime(file_path) > FILE_RETENTION:
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        print(f"Error cleaning up {filename}: {e}")

def start_cleanup_scheduler():
    """Start the cleanup scheduler in a separate thread"""
    def scheduler():
        while True:
            time.sleep(CLEANUP_INTERVAL)
            cleanup_old_files()
    
    thread = threading.Thread(target=scheduler, daemon=True)
    thread.start()

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """Handle image upload"""
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'status': 'error', 'message': 'Both images are required'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        if not file1.filename or not file2.filename:
            return jsonify({'status': 'error', 'message': 'No files selected'}), 400

        if not (comparison_engine.allowed_file(file1.filename) and 
                comparison_engine.allowed_file(file2.filename)):
            return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400

        # Generate unique IDs
        file1_id = str(uuid.uuid4())
        file2_id = str(uuid.uuid4())

        # Save files
        file1_ext = file1.filename.rsplit('.', 1)[1].lower()
        file2_ext = file2.filename.rsplit('.', 1)[1].lower()
        
        file1_path = os.path.join(UPLOAD_FOLDER, f"{file1_id}.{file1_ext}")
        file2_path = os.path.join(UPLOAD_FOLDER, f"{file2_id}.{file2_ext}")

        file1.save(file1_path)
        file2.save(file2_path)

        # Store metadata
        temp_files[file1_id] = {
            'path': file1_path,
            'original_name': file1.filename,
            'upload_time': datetime.now()
        }
        temp_files[file2_id] = {
            'path': file2_path,
            'original_name': file2.filename,
            'upload_time': datetime.now()
        }

        return jsonify({
            'status': 'success',
            'message': 'Images uploaded successfully',
            'file_ids': [file1_id, file2_id]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_images():
    """Handle image comparison"""
    try:
        data = request.json
        file_ids = data.get('file_ids', [])
        
        if len(file_ids) != 2:
            return jsonify({'status': 'error', 'message': 'Two file IDs required'}), 400

        file1_id, file2_id = file_ids

        if file1_id not in temp_files or file2_id not in temp_files:
            return jsonify({'status': 'error', 'message': 'Files not found'}), 404

        file1_path = temp_files[file1_id]['path']
        file2_path = temp_files[file2_id]['path']

        # Get ROI parameters if provided
        roi1 = data.get('roi1')  # [x, y, width, height]
        roi2 = data.get('roi2')

        # Process comparison
        result = comparison_engine.process_comparison(file1_path, file2_path, roi1, roi2)

        # Save result images
        result_id = str(uuid.uuid4())
        comparison_path = os.path.join(RESULTS_FOLDER, f"comparison_{result_id}.png")
        highlighted_path = os.path.join(RESULTS_FOLDER, f"highlighted_{result_id}.png")
        original_path = os.path.join(RESULTS_FOLDER, f"original_{result_id}.png")

        cv2.imwrite(comparison_path, result['final_comparison'])
        cv2.imwrite(highlighted_path, result['highlighted_with_text'])
        cv2.imwrite(original_path, result['region1_original'])

        # Prepare classification summary
        classifications = {}
        for r in result['results']:
            cls = r['classification']
            classifications[cls] = classifications.get(cls, 0) + 1

        return jsonify({
            'status': 'success',
            'results': {
                'differences_found': len(result['results']),
                'classifications': classifications,
                'output_images': {
                    'comparison': f"/api/results/comparison_{result_id}.png",
                    'highlighted': f"/api/results/highlighted_{result_id}.png",
                    'original': f"/api/results/original_{result_id}.png"
                },
                'details': result['results']
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/results/<filename>')
def serve_result(filename):
    """Serve result images"""
    try:
        return send_from_directory(RESULTS_FOLDER, filename)
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Start cleanup scheduler
    start_cleanup_scheduler()