# 🧮 Handwritten Equation Solver — Web App

A Streamlit web application for recognizing and solving handwritten math equations using CNN + SymPy.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Model File Exists
Make sure `symbol_cnn_v2.pth` is in the same directory as `app.py`.

### 3. Run the App
```bash
streamlit run app.py
```

Opens in browser at `http://localhost:8501`

---

## 📝 How to Use

1. **Upload** a handwritten equation image (JPG or PNG)
2. **View** segmentation — each symbol boxed (🟢 normal, 🟠 exponent)
3. **Check** symbol recognition with confidence scores
4. **Correct** any wrong predictions in the correction table
5. **Get** the solved equation instantly

---

## 📊 Supported Features

| Type | Example | Result |
|---|---|---|
| Arithmetic | `3 + 5 =` | 8 |
| Linear equation | `3x + 5 = 11` | x = 2 |
| Quadratic | `x² + 5x + 6 = 0` | x = -2, -3 |
| Brackets | `2(x + 1) = 8` | x = 3 |
| Exponents | `x² + 3 = 12` | x = 3 |
| Division | `12 ÷ 4 =` | 3 |

**Supported symbols:** `0–9`, `+`, `-`, `×`, `÷`, `=`, `x`, `(`, `)`, `^`

---

## 🔁 Active Learning — Correction & Retraining

The app supports user-guided model improvement:

1. **Correction table** — edit any wrong predicted symbol
2. **Save corrections** — crops saved to `corrections/` with label
3. **Export** — crops organized into class folders for retraining
4. **Retrain** — one-click fine-tuning on corrected examples
5. Backup of previous model auto-saved before each retrain

---

## 🎨 Tips for Best Results

- White background, black ink, thick strokes
- Clear spacing between symbols — segmentation relies on gaps
- Write **exponents small and raised** above the baseline
- Target ~60px height for the full equation
- For brackets: write tall and curved, not wide

---

## 📁 File Structure

```
handwritten_equation_solver/
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
├── symbol_cnn_v2.pth                  # Trained CNN model (16 classes)
├── symbol_cnn_v2_backup_<ts>.pth      # Auto-backups before retraining
├── handwritten_equation_solver_v2.ipynb  # Training notebook
├── corrections/                        # User correction data
│   ├── correction_log.csv             # Log of all corrections
│   ├── correction_<ts>_<n>_<sym>.png  # Saved symbol crops
│   └── retrain_dataset/               # Organized for retraining
│       ├── +/
│       ├── -/
│       └── ...
├── data/
│   ├── digits/    # olafkrastovski dataset
│   └── math/      # sagyamthapa dataset
└── README_WEB_APP.md
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Ensure `symbol_cnn_v2.pth` is in same folder as `app.py` |
| `Shape mismatch` error | Don't change `CNN_NUM_CLASSES` — must stay 16 to match saved weights |
| Poor digit recognition | Use the correction table to fix and retrain |
| Exponent not detected | Write exponent clearly smaller and raised above baseline |
| Bracket not recognized | Bracket detected by shape (tall + curved) — write clearly |
| Solver error | Check parsed expression in Debug Info expander |

---

## 🧠 Model Architecture

- **Type:** LeNet-inspired CNN (3 conv blocks + fully connected)
- **Input:** 32×32 grayscale image per symbol
- **Output:** 16 classes (digits 0–9, +, -, ×, ÷, =, x)
- **Brackets `(` `)` and exponents `^`:** detected via shape/position heuristics, not CNN
- **Framework:** PyTorch

---

## 🚀 Deployment Options

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → deploy

---

**Enjoy solving handwritten equations! ✏️➕🎯**
