# TireDetector Evaluation Test Specification

## Overview
This document specifies the evaluation tests for the TireDetector model. Unlike traditional unit tests, these are ML evaluation tests that assess model performance on various scenarios.

## Test Categories

### 1. Positive Tests (15 images)
**Goal:** Verify the model detects tires when they ARE present

**Success Criteria:**
- At least 1 detection per image
- Confidence >= 0.6 (configurable threshold)
- Bounding boxes are reasonable (width/height > 10px)

**Test Cases:**
- `test_positive_images_detect_tires()` - Batch test all positive images
- `test_positive_single_high_confidence()` - Sample 1 image, expect high confidence (>0.7)

**Expected Behavior:**
- Should detect: 15/15 images (100%)
- Average confidence: >0.7

---

### 2. Negative Tests (10 images)
**Goal:** Verify the model does NOT detect tires when they are absent

**Success Criteria:**
- Zero detections OR all detections below threshold (0.6)

**Test Cases:**
- `test_negative_images_no_false_positives()` - Batch test all negative images

**Expected Behavior:**
- Should NOT detect: 10/10 images (100%)
- False positive rate: 0%

**Images:**
- Grass, portraits, abstract art, indoor scenes

---

### 3. Edge Cases - False Positive Risk (5 images)
**Goal:** Test model's ability to distinguish tire-like circular objects from actual tires

**Success Criteria:**
- Should NOT detect tires in these images
- Model should resist false positives from circular shapes

**Test Cases:**
- `test_frisbee_not_detected_as_tire()` - Frisbees (08, 09)
- `test_donuts_not_detected_as_tire()` - Donuts (10)
- `test_cdj_equipment_not_detected_as_tire()` - CDJ turntables (06, 07)

**Expected Behavior:**
- Should NOT detect: 5/5 images
- False positive rate: 0%

---

### 4. Edge Cases - Difficult Positives (5 images)
**Goal:** Test model performance on challenging but valid tire images

**Success Criteria (Relaxed):**
- May have lower confidence (>= 0.4)
- May detect fewer tires than actually present
- Acceptance: At least 60% detection rate

**Test Cases:**
- `test_motion_blur_tire_detection()` - Motion blur (04)
- `test_blurry_car_tire_detection()` - Blurry car (03)
- `test_distant_vehicle_tire_detection()` - Far away vehicles (01)
- `test_flat_tire_detection()` - Flat/unusual tire (05)
- `test_busy_scene_tire_detection()` - Multiple vehicles (02)

**Expected Behavior:**
- Should detect: 3/5 images minimum (60%)
- Average confidence may be lower (0.4-0.6)

---

## Test Metrics to Track

### Precision Metrics
```
True Positives (TP): Positive images correctly detected
False Positives (FP): Negative/false-positive-risk images incorrectly detected
False Negatives (FN): Positive images NOT detected
True Negatives (TN): Negative images correctly NOT detected
```

### Calculated Metrics
```
Precision = TP / (TP + FP)  # How many detections are correct?
Recall = TP / (TP + FN)     # How many actual tires did we find?
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

### Target Performance
- Precision: >= 90%
- Recall: >= 80%
- F1 Score: >= 0.85

---

## Test Configuration

### Confidence Thresholds
```python
CONFIDENCE_THRESHOLD_HIGH = 0.7   # For clear positive images
CONFIDENCE_THRESHOLD_MEDIUM = 0.6  # Default threshold
CONFIDENCE_THRESHOLD_LOW = 0.4    # For edge case positives
```

### Bounding Box Validation
```python
MIN_BBOX_SIZE = 10  # pixels
```

---

## Test Implementation Approach

### Jest-style comparison:
```python
# Traditional Jest (deterministic)
expect(add(2, 2)).toBe(4)

# ML Eval (probabilistic)
assert len(detections) > 0
assert detections[0]['confidence'] >= 0.6
```

### Parametrized Tests
Use `pytest.mark.parametrize` to run the same test logic across all images in a category.

### Test Fixtures
```python
@pytest.fixture
def detector():
    return TireDetector()

@pytest.fixture
def positive_images():
    return load_images_from_dir('fixtures/images/positive')
```

---

## Running Tests

```bash
# Run all tests
pytest tests/test_detector.py -v

# Run specific category
pytest tests/test_detector.py -k "positive"
pytest tests/test_detector.py -k "negative"
pytest tests/test_detector.py -k "edge_case"

# Run with coverage
pytest tests/test_detector.py --cov=services.models.detector
```

---

## Baseline Results Template

After first run, document baseline:

```
## Baseline Results (YYYY-MM-DD)

### Positive Tests
- Detection rate: X/15 (X%)
- Average confidence: 0.XX
- Failed images: [list]

### Negative Tests
- True negatives: X/10 (X%)
- False positives: X/10 (X%)
- Failed images: [list]

### Edge Cases - False Positive Risk
- True negatives: X/5 (X%)
- False positives: [list]

### Edge Cases - Difficult Positives
- Detection rate: X/5 (X%)
- Average confidence: 0.XX

### Overall Metrics
- Precision: X.XX
- Recall: X.XX
- F1 Score: X.XX
```
