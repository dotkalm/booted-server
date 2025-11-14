"""
Evaluation tests for TireDetector.

Unlike traditional unit tests, these are ML evaluation tests that assess
model performance on various image scenarios with probabilistic outputs.
"""

import pytest
from pathlib import Path
from PIL import Image
from services.models.detector import TireDetector

# Test configuration
CONFIDENCE_THRESHOLD_HIGH = 0.7
CONFIDENCE_THRESHOLD_MEDIUM = 0.6
CONFIDENCE_THRESHOLD_LOW = 0.4
MIN_BBOX_SIZE = 10

# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
POSITIVE_DIR = FIXTURES_DIR / "positive"
NEGATIVE_DIR = FIXTURES_DIR / "negative"
EDGE_CASES_DIR = FIXTURES_DIR / "edge_cases"


@pytest.fixture(scope="module")
def detector():
    """Initialize detector once for all tests."""
    return TireDetector()


@pytest.fixture
def positive_images():
    """Load all positive test images."""
    return list(POSITIVE_DIR.glob("*.png"))


@pytest.fixture
def negative_images():
    """Load all negative test images."""
    return list(NEGATIVE_DIR.glob("*.png"))


# ============================================================================
# POSITIVE TESTS - Should detect tires
# ============================================================================

def test_positive_images_detect_tires(detector, positive_images):
    """Test that model detects tires in all positive images."""
    detections_count = 0
    failed_images = []

    for img_path in positive_images:
        image = Image.open(img_path)
        detections = detector.detect_tires(image)

        if len(detections) > 0 and detections[0]['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM:
            detections_count += 1
        else:
            failed_images.append(img_path.name)

    detection_rate = detections_count / len(positive_images)

    print(f"\nPositive Detection Rate: {detections_count}/{len(positive_images)} ({detection_rate:.1%})")
    if failed_images:
        print(f"Failed images: {failed_images}")

    # Expect at least 80% detection rate
    assert detection_rate >= 0.8, f"Detection rate {detection_rate:.1%} is below 80%"


@pytest.mark.parametrize("image_file", [
    "01.png", "05.png", "10.png"
])
def test_positive_sample_high_confidence(detector, image_file):
    """Test that clear tire images have high confidence detections."""
    img_path = POSITIVE_DIR / image_file
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    assert len(detections) > 0, f"No tires detected in {image_file}"
    assert detections[0]['confidence'] >= CONFIDENCE_THRESHOLD_HIGH, \
        f"Confidence {detections[0]['confidence']:.3f} is below {CONFIDENCE_THRESHOLD_HIGH}"
    assert detections[0]['class'] == 'tire'

    # Validate bounding box is reasonable
    bbox = detections[0]['bbox']
    assert bbox['width'] >= MIN_BBOX_SIZE, "Bounding box too small"
    assert bbox['height'] >= MIN_BBOX_SIZE, "Bounding box too small"


# ============================================================================
# NEGATIVE TESTS - Should NOT detect tires
# ============================================================================

def test_negative_images_no_false_positives(detector, negative_images):
    """Test that model does not detect tires in negative images."""
    true_negatives = 0
    false_positives = []

    for img_path in negative_images:
        image = Image.open(img_path)
        detections = detector.detect_tires(image)

        # Filter detections by confidence threshold
        high_conf_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM]

        if len(high_conf_detections) == 0:
            true_negatives += 1
        else:
            false_positives.append({
                'image': img_path.name,
                'detections': high_conf_detections
            })

    true_negative_rate = true_negatives / len(negative_images)

    print(f"\nTrue Negative Rate: {true_negatives}/{len(negative_images)} ({true_negative_rate:.1%})")
    if false_positives:
        print(f"False positives: {[fp['image'] for fp in false_positives]}")

    # Expect at least 90% true negative rate (low false positive rate)
    assert true_negative_rate >= 0.9, f"Too many false positives: {len(false_positives)}"


# ============================================================================
# EDGE CASES - False Positive Risk (tire-like circular objects)
# ============================================================================

@pytest.mark.parametrize("image_file,description", [
    ("08_frisbee.png", "Frisbee"),
    ("09_frisbee.png", "Frisbee 2"),
    ("10_donuts.png", "Donuts"),
])
def test_circular_objects_not_detected_as_tires(detector, image_file, description):
    """Test that circular tire-like objects are not detected as tires."""
    img_path = EDGE_CASES_DIR / image_file
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    high_conf_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM]

    assert len(high_conf_detections) == 0, \
        f"{description} incorrectly detected as tire with confidence {high_conf_detections[0]['confidence']:.3f}"


@pytest.mark.parametrize("image_file,description", [
    ("06_cdj.png", "CDJ Equipment"),
    ("07_cdj.png", "CDJ Equipment 2"),
])
def test_cdj_equipment_not_detected_as_tires(detector, image_file, description):
    """Test that CDJ/DJ turntables are not detected as tires."""
    img_path = EDGE_CASES_DIR / image_file
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    high_conf_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM]

    assert len(high_conf_detections) == 0, \
        f"{description} incorrectly detected as tire"


# ============================================================================
# EDGE CASES - Difficult Positives (challenging but valid tire images)
# ============================================================================

@pytest.mark.parametrize("image_file,description", [
    ("04_car_motion_blur.png", "Motion blur car"),
    ("03_car_blurry.png", "Blurry car"),
])
def test_blurry_tire_detection(detector, image_file, description):
    """Test detection on blurry/motion blur images (relaxed threshold)."""
    img_path = EDGE_CASES_DIR / image_file
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    # Use lower threshold for difficult cases
    valid_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_LOW]

    # It's OK if these don't always detect, but document it
    if len(valid_detections) > 0:
        print(f"\n{description}: Detected with confidence {valid_detections[0]['confidence']:.3f}")
    else:
        print(f"\n{description}: No detection (acceptable for difficult case)")


def test_distant_vehicle_tire_detection(detector):
    """Test detection on distant/small vehicles."""
    img_path = EDGE_CASES_DIR / "01_cars.png"
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    if len(detections) > 0:
        print(f"\nDistant vehicles: {len(detections)} tires detected")
        print(f"   Confidences: {[d['confidence'] for d in detections]}")
    else:
        print(f"\nDistant vehicles: No detection (acceptable - tires very small)")


def test_flat_tire_detection(detector):
    """Test detection on flat/unusual tire."""
    img_path = EDGE_CASES_DIR / "05_car_flat_tire.png"
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    valid_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_LOW]

    if len(valid_detections) > 0:
        print(f"\nFlat tire: Detected with confidence {valid_detections[0]['confidence']:.3f}")
    else:
        print(f"\nFlat tire: No detection (acceptable - unusual appearance)")


def test_busy_scene_tire_detection(detector):
    """Test detection in busy scenes with multiple vehicles."""
    img_path = EDGE_CASES_DIR / "02_buses.png"
    image = Image.open(img_path)
    detections = detector.detect_tires(image)

    if len(detections) > 0:
        print(f"\nBusy scene: {len(detections)} tires detected")
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        print(f"   Average confidence: {avg_conf:.3f}")
    else:
        print(f"\nBusy scene: No detection")


# ============================================================================
# SUMMARY METRICS TEST
# ============================================================================

def test_overall_performance_metrics(detector, positive_images, negative_images):
    """Calculate and report overall performance metrics."""
    # True Positives: Positive images correctly detected
    tp = 0
    for img_path in positive_images:
        image = Image.open(img_path)
        detections = detector.detect_tires(image)
        if len(detections) > 0 and detections[0]['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM:
            tp += 1

    # False Negatives: Positive images NOT detected
    fn = len(positive_images) - tp

    # True Negatives: Negative images correctly NOT detected
    tn = 0
    for img_path in negative_images:
        image = Image.open(img_path)
        detections = detector.detect_tires(image)
        high_conf = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD_MEDIUM]
        if len(high_conf) == 0:
            tn += 1

    # False Positives: Negative images incorrectly detected
    fp = len(negative_images) - tn

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print("OVERALL PERFORMANCE METRICS")
    print("="*60)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")
    print("-"*60)
    print(f"Precision: {precision:.2%} (target >= 90%)")
    print(f"Recall:    {recall:.2%} (target >= 80%)")
    print(f"F1 Score:  {f1_score:.3f} (target >= 0.85)")
    print("="*60)

    # Assert target metrics
    assert precision >= 0.90, f"Precision {precision:.2%} below target (90%)"
    assert recall >= 0.80, f"Recall {recall:.2%} below target (80%)"
    assert f1_score >= 0.85, f"F1 Score {f1_score:.3f} below target (0.85)"
