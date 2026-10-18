# 👞 Khybersole — AI Foot Size Finder & Heritage Chappal Sizing

An AI-powered computer vision and photogrammetry application for **Khybersole**, an e-commerce platform specializing in authentic handcrafted Peshawari Chappals.

This system estimates customer foot lengths in millimeters using a custom-trained **YOLOv8 instance segmentation model** (`best.pt`) combined with physical reference object calibration (standard credit/debit/ID cards or A4 paper), and maps the measurement to recommended Peshawari chappal sizes.

---

## 🌟 Key Features

1. **Custom Model Foot Segmentation**: Powered by your trained YOLOv8 instance segmentation model (`best.pt`), producing binary foot masks at high confidence (~93%).
2. **Anatomical Endpoint Extraction**: Principal Component Analysis (PCA) combined with width-gradient heuristics to distinguish heel calcaneus vs. forefoot/toes and determine the true longitudinal measurement axis.
3. **Reference Object Metric Calibration**: 
   - Standard bank / ID card (ISO/IEC 7810 ID-1: 85.60 mm × 53.98 mm) or A4 paper calibration.
   - Perspective Homography mapping from camera image coordinates directly into metric millimeter space.
   - Manual 4-corner override tool for challenging lighting or low-contrast cards.
4. **Configurable Peshawari Chappal Sizing**: JSON-driven size chart (`config/size_chart.json`) covering traditional Pakistani/UK, EU, and US sizing with boundary transition advisories and fit allowances.
5. **Quality Control & Image Validation**: Automatic detection of motion blur (Laplacian variance), under/over-exposure, partial foot boundary clipping, and physical plausibility validation (180 mm – 330 mm).
6. **Privacy & Offline Execution**: Runs 100% locally on CPU without requiring paid external cloud APIs or storing customer photos remotely.

---

## 📁 Project Structure

```text
d:/FYP/model/
├── app.py                      # Interactive Streamlit customer sizing application
├── diagnose_model.py           # Model inspection & verification script
├── evaluate_foot_finder.py     # Benchmark evaluation runner on test dataset
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Complete documentation & usage guide
├── .gitignore                  # Git tracking rules
├── models/
│   └── best.pt                 # Trained YOLOv8 segmentation checkpoint (6.7 MB)
├── src/
│   ├── __init__.py
│   ├── model_loader.py         # Singleton model loader & device manager
│   ├── foot_segmentation.py    # YOLOv8 segmentation wrapper & overlay rendering
│   ├── foot_landmarks.py       # PCA longitudinal axis & anatomical endpoint extractor
│   ├── calibration.py          # Reference card detector, homography & scale calculator
│   ├── measurement.py          # Physical dimension calculator & quality scoring
│   ├── size_recommender.py     # Peshawari chappal sizing engine & boundary analysis
│   └── image_validation.py     # Image quality, blur, and lighting validator
├── config/
│   ├── size_chart.json         # Configurable Peshawari chappal sizing parameters
│   └── settings.json           # Default thresholds and reference dimensions
├── tests/
│   ├── test_calibration.py
│   ├── test_error_handling.py
│   ├── test_foot_landmarks.py
│   ├── test_image_validation.py
│   ├── test_measurement.py
│   ├── test_model_and_segmentation.py
│   └── test_size_recommender.py
├── test_images/                # Sample dataset images for local validation
└── outputs/                    # Annotated diagnostic & test output images
```

---

## 🚀 Quick Start (Windows 11 Setup)

### 1. Open Terminal in Project Folder
Open PowerShell or Windows Command Prompt:
```powershell
cd d:\FYP\model
```

### 2. Set Up Python Virtual Environment (Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Ensure Model Weights are in Place
Confirm that `best.pt` is inside `models/best.pt`:
```powershell
Test-Path models\best.pt
```

---

## 🧪 Testing & Diagnostics

### Run the Model Diagnostic Tool
Verifies model architecture, task compatibility, class names, and produces a test prediction:
```powershell
python diagnose_model.py
```

### Run the Automated Unit Test Suite
Runs all 22 test cases covering calibration, landmarks, sizing, error handling, and image validation:
```powershell
python -m pytest tests/ -v
```

### Run the Dataset Benchmark Evaluation
Evaluates all test images in `test_images/` and prints detection rates, confidence, latency, and aspect ratio metrics:
```powershell
python evaluate_foot_finder.py
```

---

## 💻 Running the Streamlit Web Application

Launch the interactive web interface:
```powershell
streamlit run app.py
```
*(Or `python -m streamlit run app.py`)*

Open your browser at:
👉 **`http://localhost:8501`**

---

## 📷 Photography Guidelines for Accurate Measurements

For best measurement results:
1. **Surface**: Place bare foot on a flat, solid floor (wood, tile, or plain paper).
2. **Card Placement**: Lay a standard bank/credit card flat on the floor directly next to the foot.
3. **Camera Angle**: Hold camera directly above the foot pointing straight down (perpendicular to the floor) to avoid foreshortening.
4. **Lighting**: Ensure even ambient lighting without harsh flash glares or heavy cast shadows.

---

## ⚙️ Configuring the Peshawari Chappal Size Chart

To update or adjust size ranges to match your specific footwear lasts, edit `config/size_chart.json`. No Python code edits are required:
```json
{
  "size_pk_uk": 8,
  "size_eu": 41,
  "size_us": 9,
  "min_foot_length_mm": 257.1,
  "max_foot_length_mm": 266.0,
  "recommended_foot_length_mm": 261.0,
  "label": "Size 8 (UK / PK) — Medium"
}
```

---

## 🔧 Troubleshooting

- **Error: `ModelNotFoundError: Could not locate foot segmentation weights (best.pt)`**
  * Place `best.pt` in `models/best.pt` or in the root `d:\FYP\model` folder.
- **Reference card not detected automatically**:
  * In the app sidebar, check **"Manual Card Corner Override"** to specify the card corners directly.
- **Plausibility Warning: `Calculated foot length is below/above physiological bounds`**:
  * Verify that the reference card was correctly detected and that the camera angle was top-down.
