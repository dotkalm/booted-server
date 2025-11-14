# Baseline Test Results

**Date:** 2025-11-13
**Model:** yolov8n.pt (pretrained COCO)

## Critical Finding

**The model is NOT detecting tires!**

### Root Cause
The current model (`yolov8n.pt`) is a standard YOLO model trained on the COCO dataset, which includes 80 classes like:
- person, car, truck, bicycle, etc.

**BUT COCO does NOT include "tire" as a class.**

Your `TireDetector` code filters for `class_id == 0` (services/models/detector.py:42), which in COCO is the "person" class, not "tire".

### What the model IS detecting:
- Cars, trucks, persons, bicycles (COCO classes)
- Frisbees, boats, TVs, keyboards, etc.

### What the model is NOT detecting:
- **Tires** (not in COCO dataset)

---

## Test Results Summary

### Positive Tests (Should detect tires)
- **Detection rate:** 1/15 (6.7%) ❌ FAIL
- **Target:** >= 80%
- **Failed images:** 14 out of 15
- **Issue:** Model detects cars/trucks but filters them out because class_id != 0

### Negative Tests (Should NOT detect tires)
- **True negative rate:** 8/10 (80.0%)
- **False positives:** 2 images (06.png, 08.png)
- **Target:** >= 90% (MISSED by 10%)

### Edge Cases - Circular Objects
- **Frisbee (08):** PASSED (not detected as tire)
- **Frisbee (09):** PASSED (not detected as tire)
- **Donuts (10):** PASSED (not detected as tire)

### Edge Cases - CDJ Equipment
- **CDJ (06):** PASSED (not detected as tire)
- **CDJ (07):** PASSED (not detected as tire)

### Edge Cases - Difficult Positives
- **Motion blur car (04):** No detection
- **Blurry car (03):** No detection
- **Distant vehicles (01):** No detection
- **Flat tire (05):** No detection
- **Busy scene (02):** No detection

---

## Overall Performance Metrics

```
True Positives (TP):   1
False Positives (FP):  2
True Negatives (TN):   8
False Negatives (FN): 14
```

**Calculated Metrics:**
- **Precision:** 33.33% (target >= 90%) ❌ FAIL
- **Recall:** 6.67% (target >= 80%) ❌ FAIL
- **F1 Score:** 0.111 (target >= 0.85) ❌ FAIL

---

## Recommendations

### Option 1: Use Custom Trained Tire Detection Model
You need a YOLO model trained specifically on tire images. Your `WHEEL22` directory suggests you may have training data.

**Steps:**
1. Train YOLOv8 on tire dataset
2. Replace `yolov8n.pt` with custom model
3. Update `class_id == 0` to match your tire class ID

### Option 2: Detect Vehicles Instead
If you want to detect cars/trucks (which contain tires), change the filter:

```python
# In detector.py line 42, change:
if class_id == 0:  # person class

# To:
if class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
```

### Option 3: Train on Existing Data
Use the training images in `models/WHEEL22/` to train a custom tire detector.

---

## Next Steps

1. **Decide detection target:** Tires specifically OR vehicles containing tires?
2. **Train custom model** if targeting tires
3. **Adjust class filtering** in `detector.py:42`
4. **Re-run eval tests** after changes
5. **Iterate** until metrics meet targets (Precision >= 90%, Recall >= 80%)
