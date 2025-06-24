import cv2
import numpy as np
from scipy.stats import alpha
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Supports many languages

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

    max_dim = 1500
    scale = 1.0
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    clone = image.copy()
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, draw_rectangle, param=(clone, window_name))
    print(f"Select a region in {window_name}, then press any key.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if len(roi_points) != 2:
        raise ValueError("ROI not selected properly")

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
    # Create a copy of the image to work with
    result_image = image.copy()

    # Create a mask for text regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Get image dimensions
    height, width = image.shape[:2]

    text_boxes = []  # To store text box coordinates and their corresponding text

    # Process the image in tiles
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Calculate the tile boundaries
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)

            # Extract the tile
            tile = image[y:end_y, x:end_x]

            # Run PaddleOCR on the tile
            # result = ocr.ocr(tile, cls=True)
            result = ocr.predict(tile)

            # Check if result is valid
            if result is not None and len(result) > 0:
                for idx in range(len(result)):
                    if result[idx] is None:
                        continue
                    for line in result[idx]:
                        try:
                            if len(line) >= 1:  # Ensure there's a box
                                box = line[0]
                                if isinstance(box, list) and len(box) == 4:  # Ensure it's a valid box with 4 points
                                    # Adjust box coordinates to the original image
                                    adjusted_box = [[point[0] + x, point[1] + y] for point in box]

                                    text_boxes.append((adjusted_box, line[1]))  # Store box and text

                                    # Convert box points to integer array for drawing
                                    pts = np.array(adjusted_box).astype(np.int32).reshape((-1, 1, 2))

                                    # Fill the text region in the mask
                                    cv2.fillPoly(mask, [pts], 255)
                        except Exception as e:
                            print(f"Error processing detection: {e}")

    # Apply dilation to the mask to better cover text regions
    if np.max(mask) > 0:  # Only if something was detected
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply inpainting with a larger radius for better results
    inpainted = cv2.inpaint(result_image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    cv2.imwrite("text_mask.png", mask)

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
        # Use default SSIM
        score, _ = ssim(gray1, gray2, full=True)
        similarity = score

    return similarity < threshold


def classify_difference(patch1, patch2):
    # If patch2 is mostly blank/white, it's missing
    if is_blank(patch2, threshold=10):
        return "missing"
    # Otherwise check similarity
    elif is_modified(patch1, patch2, threshold=0.6):
        return "modified"
    else:
        return "unchanged"

def compare_regions(region1, region2):
    region2_aligned, H = align_with_orb(region1, region2)
    # Convert to grayscale
    gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(region2_aligned, cv2.COLOR_BGR2GRAY)

    # Detect horizontal lines using horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_h1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detect_h2 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Detect vertical lines using vertical kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_v1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    detect_v2 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Combine horizontal and vertical line detections
    lines1 = cv2.add(detect_h1, detect_v1)
    lines2 = cv2.add(detect_h2, detect_v2)

    # Find line differences
    line_diff = cv2.absdiff(lines1, lines2)

    # Normal difference processing
    normal_diff = cv2.absdiff(gray1, gray2)

    # Combine both differences with higher weight for line differences
    diff = cv2.addWeighted(normal_diff, 0.5, line_diff, 1.0, 0)

    # After calculating the diff and normalizing
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Apply Gaussian blur to reduce noise before thresholding
    blurred_diff = cv2.GaussianBlur(norm_diff, (5, 5), 0)

    # Use more conservative adaptive thresholding parameters
    thresh_diff = cv2.adaptiveThreshold(
        blurred_diff,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,  # Larger block size
        7  # Higher constant to reduce sensitivity
    )

    # Apply an additional global threshold to filter out very low differences
    _, global_mask = cv2.threshold(norm_diff, 25, 255, cv2.THRESH_BINARY)
    thresh_diff = cv2.bitwise_and(thresh_diff, global_mask)

    # Very minimal morphological operations to preserve thin lines
    kernel = np.ones((2, 2), np.uint8)

    # Skip opening entirely
    dilated = cv2.dilate(thresh_diff, kernel, iterations=1)

    # Find contours with minimal filtering
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_cleaned = np.zeros_like(dilated)

    for cnt in contours:
        # For technical diagrams, even tiny differences can be important
        cv2.drawContours(mask_cleaned, [cnt], -1, 255, -1)

    # Create a yellow mask over differences
    color_mask = np.zeros_like(region2_aligned)
    color_mask[mask_cleaned > 0] = (0, 180, 180)
    highlighted = cv2.addWeighted(region2_aligned, 0.7, color_mask, 0.3, 0)

    # Final side-by-side view
    comparison = np.hstack([region1, highlighted])
    return region1, highlighted, comparison, mask_cleaned, region2_aligned, H


def analyze_differences(region1, mask_cleaned, region2_aligned, highlighted):
    results = []
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a clean overlay image to draw on
    overlay = highlighted.copy()
    annotated_side_by_side = np.hstack((region1.copy(), highlighted.copy()))

    for i, cnt in enumerate(contours):
        patch1, (x, y, w, h) = extract_patch(region1, cnt)
        patch2 = region2_aligned[y:y + h, x:x + w]

        classification = classify_difference(patch1, patch2)
        if classification == "unchanged":
            continue

        # Use these distinct colors for different change types
        color_map = {
            "missing": (0, 0, 255),  # Red (BGR)
            "modified": (0, 255, 0),  # Green (BGR)
            "unchanged": (0, 180, 180)  # Yellow-ish
        }
        color = color_map.get(classification, (0, 180, 180))

        # Create a mask for this contour
        contour_mask = np.zeros_like(overlay)
        cv2.drawContours(contour_mask, [cnt], -1, color, thickness=cv2.FILLED)

        # Blend this contour mask onto the overlay with proper alpha
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1.0, contour_mask, alpha, 0)

        # Draw the same contour on the right side of the side-by-side view
        offset = region1.shape[1]
        right_contour_mask = np.zeros_like(region1)
        cv2.drawContours(right_contour_mask, [cnt], -1, color, thickness=cv2.FILLED)

        # Blend into the right side
        mask_region = annotated_side_by_side[:, offset:]
        blended_region = cv2.addWeighted(mask_region, 1.0, right_contour_mask, alpha, 0)
        annotated_side_by_side[:, offset:] = blended_region

        # Add to results
        results.append({
            "box": (x, y, w, h),
            "classification": classification
        })

    # Add legend and return
    annotated_side_by_side = draw_legend(annotated_side_by_side)
    return results, overlay, annotated_side_by_side

def draw_legend(image):
    legend_items = [
        ("Missing", (0, 0, 255)),  # Red
        ("Modified", (0, 255, 0)),  # Green
        ("Other", (0, 255, 255))  # Yellow
    ]

    start_x, start_y = 10, image.shape[0] - 100
    box_size = 20
    spacing = 35

    for i, (label, color) in enumerate(legend_items):
        y = start_y + i * spacing
        cv2.rectangle(image, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        cv2.putText(image, label, (start_x + box_size + 10, y + box_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    return image


def create_text_mask(text_boxes, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for box, _ in text_boxes:
        # Convert box to numpy array of points
        points = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        # Fill the polygon
        cv2.fillPoly(mask, [points], 255)

    return mask


def main(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error loading images.")
        return

    # Select ROI on the first image
    x1, y1, w1, h1 = select_roi_on_image(image1)
    x2, y2, w2, h2 = select_roi_on_image(image2)

    # Extract the selected region from both images
    region1_original = image1[y1:y1 + h1, x1:x1 + w1].copy()
    region2_original = image2[y2:y2 + h2, x2:x2 + w2].copy()

    # Extract text from both images
    print("Detecting text in first image...")
    region1_no_text, text_mask1, text_boxes1 = remove_text_with_paddleocr(region1_original)

    print("Detecting text in second image...")
    region2_no_text, text_mask2, text_boxes2 = remove_text_with_paddleocr(region2_original)

    # Compare regions without text to identify drawing differences only
    print("Comparing regions (drawings only)...")
    region1_clean, highlighted_clean, comparison_clean, mask_cleaned, region2_aligned_clean, H = compare_regions(
        region1_no_text, region2_no_text)

    # Align the original second image (with text) to match the first image
    region2_original_aligned = cv2.warpPerspective(
        region2_original, H, (region1_original.shape[1], region1_original.shape[0]))

    # Analyze differences in drawings
    print("Analyzing differences...")
    results, overlay_clean, annotated_comparison_clean = analyze_differences(
        region1_clean, mask_cleaned, region2_aligned_clean, highlighted_clean)

    # Create a mask for the differences in drawings only
    drawing_diff_mask = np.zeros(region1_original.shape[:2], dtype=np.uint8)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        classification = classify_difference(
            extract_patch(region1_clean, cnt)[0],
            extract_patch(region2_aligned_clean, cnt)[0])

        if classification != "unchanged":
            cv2.drawContours(drawing_diff_mask, [cnt], -1, 255, -1)

    # Create a color mask for the differences (yellow)
    color_mask = np.zeros_like(region1_original)
    color_mask[drawing_diff_mask > 0] = (0, 255, 255)  # Yellow for differences

    # Apply the drawing differences highlight to the original images with text
    highlighted_with_text = cv2.addWeighted(region2_original_aligned, 0.7, color_mask, 0.3, 0)

    # Create the final side-by-side comparison with original text and drawing highlights
    final_comparison = np.hstack([region1_original, highlighted_with_text])
    final_comparison = draw_legend(final_comparison)

    print("Saving results...")
    cv2.imwrite("comparison_original_with_highlights.png", final_comparison)
    cv2.imwrite("image1_original.png", region1_original)
    cv2.imwrite("image2_aligned_with_highlights.png", highlighted_with_text)

    print("Done! Results saved.")


# === Run it ===
if __name__ == "__main__":
    main("Old/1.png", "New/1.png")