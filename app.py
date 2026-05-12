"""
Handwritten Equation Solver - Streamlit Web App

Full pipeline: Upload equation image → Segment → Recognize symbols → Solve
"""

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import sympy as sp
from PIL import Image
import re
import csv
import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Handwritten Equation Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CLASS DEFINITIONS & CONSTANTS
# ============================================================
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    '+', '-', 'times', 'div', '=', 'x', '(', ')', '^'
]
DISPLAY = [
    '0','1','2','3','4','5','6','7','8','9',
    '+', '-', '×', '÷', '=', 'x', '(', ')', '^'
]
CNN_NUM_CLASSES = len(CLASSES)   # 19 — matches saved symbol_cnn_v2.pth
NUM_CLASSES     = CNN_NUM_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOW_CONFIDENCE_THRESHOLD = 0.70
VALID_SYMBOLS = ['0','1','2','3','4','5','6','7','8','9','+','-','×','÷','=','x','(',')','^']
CORRECTION_LOG_DIR = Path(__file__).parent / 'corrections'
CORRECTION_LOG_DIR.mkdir(exist_ok=True)
CORRECTION_DATASET_DIR = CORRECTION_LOG_DIR / 'retrain_dataset'
CORRECTION_DATASET_DIR.mkdir(exist_ok=True)
DISPLAY_TO_CLASS = dict(zip(DISPLAY, CLASSES))

# ============================================================
# CNN MODEL ARCHITECTURE
# ============================================================
class SymbolCNN(nn.Module):
    """LeNet-inspired CNN for handwritten symbol recognition."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(SymbolCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_model():
    """Load trained model from checkpoint."""
    model_path = Path(__file__).parent / 'symbol_cnn_v2.pth'
    
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Make sure 'symbol_cnn_v2.pth' is in the same directory as this app.")
        st.stop()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SymbolCNN(num_classes=CNN_NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
def get_transform():
    """Get image preprocessing transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# ============================================================
# IMAGE PROCESSING
# ============================================================
def preprocess_image(image_path):
    """Load and binarize the equation image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')

    h, w = img.shape
    if h < 60:
        scale = 60 / h
        img = cv2.resize(img, (int(w * scale), 60))

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return img, binary


def segment_symbols(image_path, min_area=30, padding=4):
    """Segment equation image into individual symbol crops.
    Returns crops as (x, y, w, h, pil_image, is_exponent).
    is_exponent=True when symbol sits above baseline and is smaller than average.
    """
    orig, binary = preprocess_image(image_path)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    if boxes:
        avg_w = sum(w for x, y, w, h in boxes) / len(boxes)
        merge_threshold = int(avg_w * 0.2)
    else:
        merge_threshold = 10

    merged = []
    for box in boxes:
        x, y, w, h = box
        if merged and x < merged[-1][0] + merged[-1][2] + merge_threshold:
            mx, my, mw, mh = merged[-1]
            nx = min(mx, x)
            ny = min(my, y)
            nw = max(mx + mw, x + w) - nx
            nh = max(my + mh, y + h) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    # Exponent detection via baseline + height
    if merged:
        bottoms  = [y + h for (x, y, w, h) in merged]
        heights  = [h     for (x, y, w, h) in merged]
        baseline = np.median(bottoms)
        avg_h    = np.mean(heights)
    else:
        baseline = avg_h = 1

    H, W = binary.shape
    crops = []
    for (x, y, w, h) in merged:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        crop = binary[y1:y2, x1:x2]
        crop_pil = Image.fromarray(255 - crop)
        # Exponent: bottom edge well above baseline AND smaller than avg height
        is_exponent = ((y + h) < baseline - avg_h * 0.3) and (h < avg_h * 0.7)
        crops.append((x1, y1, x2 - x1, y2 - y1, crop_pil, is_exponent))

    return orig, crops


# ============================================================
# SYMBOL RECOGNITION
# ============================================================
def predict_symbol(crop_pil, model, transform, top_k=3, confidence_threshold=LOW_CONFIDENCE_THRESHOLD):
    """Predict the class of a single symbol crop and flag uncertain outputs."""
    model.eval()
    img_tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    predictions = [
        (DISPLAY[idx.item()], prob.item())
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]

    if not predictions:
        return [('?', 0.0)], predictions, True

    primary_conf = predictions[0][1]
    secondary_conf = predictions[1][1] if len(predictions) > 1 else 0.0
    is_uncertain = primary_conf < confidence_threshold or secondary_conf > primary_conf - 0.15

    if primary_conf < confidence_threshold:
        return [('?', primary_conf)], predictions, True

    return predictions, predictions, is_uncertain


def _detect_bracket(crop_pil):
    """Return '(' or ')' if crop looks like a bracket, else None."""
    img = np.array(crop_pil.convert('L'))
    h, w = img.shape
    aspect = w / (h + 1e-5)
    img_bin = (img < 128).astype(np.uint8)
    fill = np.sum(img_bin) / (h * w + 1e-5)
    if aspect < 0.6 and fill < 0.4:
        left_mass  = np.sum(img_bin[:, :w//3])
        right_mass = np.sum(img_bin[:, 2*w//3:])
        total      = np.sum(img_bin) + 1e-5
        if left_mass / total > 0.5:
            return '('
        elif right_mass / total > 0.5:
            return ')'
    return None


def _build_equation_string(symbols, is_exponents):
    """Insert ** before exponent symbols."""
    result = []
    for sym, is_exp in zip(symbols, is_exponents):
        if is_exp:
            result.append('**')
        result.append(sym)
    return ''.join(result)


def recognize_equation(image_path, model, transform):
    """Full recognition pipeline for an equation image."""
    _, crops = segment_symbols(image_path)
    symbols = []
    is_exponents = []
    all_predictions = []

    for i, (x, y, w, h, crop, is_exp) in enumerate(crops):
        # Try bracket shape detection first
        bracket = _detect_bracket(crop)
        if bracket:
            best_symbol, confidence, top3, is_uncertain = bracket, 1.0, [(bracket, 1.0)], False
        else:
            preds, top3, is_uncertain = predict_symbol(crop, model, transform, top_k=3)
            best_symbol, confidence = preds[0]

        symbols.append(best_symbol)
        is_exponents.append(is_exp)
        all_predictions.append({
            'index': i + 1,
            'symbol': best_symbol,
            'confidence': confidence,
            'top3': top3,
            'uncertain': is_uncertain,
            'is_exponent': is_exp,
            'crop': crop
        })

    equation_str = _build_equation_string(symbols, is_exponents)
    return symbols, equation_str, crops, all_predictions


# ============================================================
# EXPRESSION PARSER & SOLVER
# ============================================================
def parse_expression(symbols):
    """Convert symbols to SymPy-parseable string with brackets and exponent support."""
    if not symbols:
        raise ValueError("No symbols provided")

    sym_map = {
        '×': '*', '÷': '/', '−': '-', 'times': '*', 'div': '/',
        'sub': '-', 'add': '+', 'mul': '*', '^': '**'
    }
    tokens = [sym_map.get(s, s) for s in symbols]
    expr = ''.join(tokens)

    # ^ already mapped to ** above; handle any remaining ^
    expr = expr.replace('^', '**')

    # Implicit multiply rules
    expr = re.sub(r'(\d)([a-zA-Z(])',    r'\1*\2', expr)   # 3x, 2(
    expr = re.sub(r'([a-zA-Z])(\d)',     r'\1*\2', expr)   # x2
    expr = re.sub(r'(\))(\()',           r'\1*\2', expr)   # )(
    expr = re.sub(r'(\))([a-zA-Z\d])',   r'\1*\2', expr)   # )x, )2
    expr = re.sub(r'([a-zA-Z\d])(\()',   r'\1*\2', expr)   # x(, 2(
    expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr) # xy

    return expr.strip()


def solve_equation(symbols):
    """Parse and solve equation using SymPy with comprehensive error handling."""
    if not symbols:
        return None, "❌ No symbols to solve", ""
    
    try:
        expr_str = parse_expression(symbols)
    except Exception as e:
        return None, f"❌ Parse error: {str(e)}", ""
    
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    z = sp.Symbol('z')

    try:
        if '=' in expr_str:
            parts = expr_str.split('=')
            if len(parts) != 2:
                raise ValueError("Invalid equation format")
            
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip() if parts[1].strip() else '0'

            lhs = sp.sympify(lhs_str)
            rhs = sp.sympify(rhs_str)
            free_vars = lhs.free_symbols | rhs.free_symbols

            if not free_vars:
                # Pure arithmetic
                result = float(lhs)
                return result, f"✅ Result: **{result}**", expr_str
            else:
                # Has variables - solve for primary variable (x > y > z)
                solve_var = sorted(list(free_vars), key=lambda v: ['x', 'y', 'z'].index(str(v)) if str(v) in ['x', 'y', 'z'] else 999)[0]
                solution = sp.solve(sp.Eq(lhs, rhs), solve_var)
                if solution:
                    return solution, f"✅ Solution: {solve_var} = {solution}", expr_str
                else:
                    return None, f"⚠️ No solution found for {solve_var}", expr_str
        else:
            result = sp.simplify(sp.sympify(expr_str))
            return result, f"✅ Simplified: **{result}**", expr_str

    except sp.SympifyError as e:
        return None, f"❌ Expression error: Invalid syntax", expr_str
    except ValueError as e:
        return None, f"❌ Value error: {str(e)}", expr_str
    except Exception as e:
        return None, f"❌ Solver error: {type(e).__name__}", expr_str


def normalize_symbol_for_storage(symbol: str) -> str:
    """Convert display symbols into model class labels for storage/training."""
    if symbol is None:
        return ''
    normalized = DISPLAY_TO_CLASS.get(symbol.strip(), symbol.strip())
    return normalized


def repair_correction_log_header():
    """Repair an existing older correction log header/row format."""
    log_path = CORRECTION_LOG_DIR / 'correction_log.csv'
    if not log_path.exists():
        return

    with open(log_path, 'r', newline='', encoding='utf-8') as csvfile:
        rows = list(csv.reader(csvfile))

    if not rows:
        return

    expected_header = ['timestamp', 'index', 'predicted', 'corrected', 'corrected_label', 'top3', 'crop_path']
    old_header = ['timestamp', 'index', 'predicted', 'corrected', 'top3', 'crop_path']
    if rows[0] != old_header:
        return

    repaired = [expected_header]
    for row in rows[1:]:
        if len(row) == 7:
            timestamp, idx, predicted, corrected, top3, crop_path = row[0], row[1], row[2], row[3], row[4], row[6]
        elif len(row) == 6:
            timestamp, idx, predicted, corrected, top3, crop_path = row
        else:
            repaired.append(row)
            continue

        corrected_label = normalize_symbol_for_storage(corrected)
        repaired.append([timestamp, idx, predicted, corrected, corrected_label, top3, crop_path])

    with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(repaired)


def save_corrections_for_retraining(corrected_symbols, original_symbols, all_predictions):
    """Save corrected symbol examples for future retraining."""
    log_path = CORRECTION_LOG_DIR / 'correction_log.csv'
    timestamp = int(time.time())
    fieldnames = ['timestamp', 'index', 'predicted', 'corrected', 'corrected_label', 'top3', 'crop_path']
    saved_count = 0

    repair_correction_log_header()

    if not log_path.exists():
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for pred in all_predictions:
        idx = pred['index']
        original = original_symbols[idx - 1]
        corrected = corrected_symbols[idx - 1]
        if original == corrected:
            continue

        if corrected not in VALID_SYMBOLS:
            continue

        corrected_label = normalize_symbol_for_storage(corrected)
        if corrected_label not in CLASSES:
            continue

        base_name = sanitize_filename(f'correction_{timestamp}_{idx}_{original}_to_{corrected_label}')
        crop_name = f"{base_name}.png"
        crop_path = CORRECTION_LOG_DIR / crop_name
        pred['crop'].convert('L').save(crop_path)

        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'timestamp': timestamp,
                'index': idx,
                'predicted': original,
                'corrected': corrected,
                'corrected_label': corrected_label,
                'top3': '|'.join([f'{s}:{p:.2f}' for s,p in pred['top3']]),
                'crop_path': str(crop_path)
            })
        saved_count += 1

    return saved_count


def sanitize_filename(value: str) -> str:
    """Sanitize a string so it is safe to use in filenames."""
    safe = ''.join([c if c.isalnum() or c in ('-', '_') else '_' for c in str(value)])
    return safe.strip('_')[:255]


def export_corrections_for_retraining():
    """Export correction crops into class folders for retraining."""
    if not CORRECTION_LOG_DIR.exists():
        return 0

    log_path = CORRECTION_LOG_DIR / 'correction_log.csv'
    if not log_path.exists():
        return 0

    exported = 0
    with open(log_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            corrected = row.get('corrected', '').strip()
            corrected_label = normalize_symbol_for_storage(corrected)
            crop_path_str = row.get('crop_path', '') or row.get(None, '')
            crop_path = Path(crop_path_str)
            if not crop_path.exists():
                continue
            if corrected_label not in CLASSES:
                continue

            target_dir = CORRECTION_DATASET_DIR / corrected_label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / crop_path.name
            if not target_path.exists():
                crop = Image.open(crop_path).convert('L')
                crop.save(target_path)
                exported += 1
    return exported


def retrain_model_once(num_epochs=1, batch_size=16, lr=1e-4):
    """Retrain the loaded model once using corrected training examples."""
    if not CORRECTION_DATASET_DIR.exists():
        return 0, None, None, "No retraining dataset found."

    dataset_has_images = False
    for label_dir in CORRECTION_DATASET_DIR.iterdir():
        if label_dir.is_dir() and any(label_dir.glob("*.png")):
            dataset_has_images = True
            break

    if not dataset_has_images:
        export_corrections_for_retraining()

    samples = []
    for label_dir in CORRECTION_DATASET_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in CLASSES:
            continue
        target = CLASSES.index(label)
        for image_path in label_dir.glob("*.png"):
            samples.append((image_path, target))

    if not samples:
        return 0, None, None, "No corrected samples found for retraining."

    from torch.utils.data import Dataset, DataLoader

    class CorrectionDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert('L')
            image = self.transform(image)
            return image, label

    dataset = CorrectionDataset(samples, get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    checkpoint_path = Path(__file__).parent / 'symbol_cnn_v2.pth'
    if not checkpoint_path.exists():
        return 0, None, None, f"Model checkpoint not found: {checkpoint_path}"

    model = SymbolCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for epoch in range(num_epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    backup_path = Path(__file__).parent / f"symbol_cnn_v2_backup_{int(time.time())}.pth"
    shutil.copy2(checkpoint_path, backup_path)
    torch.save(model.state_dict(), checkpoint_path)

    return total_samples, avg_loss, accuracy, backup_path


def edit_correction_table(correction_rows):
    """Render a correction table, falling back for older Streamlit versions."""
    if hasattr(st, 'data_editor'):
        correction_df = pd.DataFrame(correction_rows)
        edited_df = st.data_editor(
            correction_df,
            num_rows="fixed",
            use_container_width=True
        )
        return edited_df['Corrected'].tolist()

    if hasattr(st, 'experimental_data_editor'):
        correction_df = pd.DataFrame(correction_rows)
        edited_df = st.experimental_data_editor(
            correction_df,
            num_rows="fixed",
            use_container_width=True
        )
        return edited_df['Corrected'].tolist()

    st.warning("Your Streamlit version does not support the table editor. Use the dropdowns below to correct each symbol.")
    corrected = []
    for row in correction_rows:
        idx = row['Symbol #']
        label = f"Symbol #{idx} - predicted {row['Predicted']}"
        options = [row['Predicted']] + [c for c in VALID_SYMBOLS if c != row['Predicted']]
        corrected_symbol = st.selectbox(label, options, index=0, key=f"manual_corr_{idx}")
        corrected.append(corrected_symbol)
    return corrected


def corrected_symbol_inputs(symbols, all_predictions):
    """Create correction widgets and return corrected symbol list."""
    corrected = symbols.copy()
    for pred in all_predictions:
        if pred.get('uncertain') or pred['symbol'] == '?':
            idx = pred['index']
            label = f"Symbol #{idx} - predicted {pred['symbol']}"
            options = [pred['symbol']] + [c for c in VALID_SYMBOLS if c != pred['symbol']]
            corrected_symbol = st.selectbox(label, options, index=0, key=f"corr_{idx}")
            corrected[idx - 1] = corrected_symbol
    return corrected


# ============================================================
# STREAMLIT UI 
# ============================================================

st.markdown("""
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    border: 1px solid #e94560;
}
.hero h1 { color: #ffffff; font-size: 2.4rem; margin: 0; }
.hero p  { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }

/* Step cards */
.step-card {
    background: #1e1e2e;
    border-left: 4px solid #e94560;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.step-card h4 { color: #e94560; margin: 0 0 0.3rem 0; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase; }
.step-card p  { color: #cdd6f4; margin: 0; font-size: 0.95rem; }

/* Solution box */
.solution-box {
    background: linear-gradient(135deg, #1a2f1a, #0d2b0d);
    border: 2px solid #40a060;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.solution-box .label { color: #80c080; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem; }
.solution-box .eq    { color: #ffffff; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
.solution-box .ans   { color: #5eff8a; font-size: 2rem; font-weight: 800; }

/* Equation display */
.eq-display {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #cba6f7;
    letter-spacing: 4px;
    margin: 0.5rem 0;
}

/* Symbol card */
.sym-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 0.6rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sym-card .sym  { font-size: 1.3rem; font-weight: 800; color: #cba6f7; }
.sym-card .conf { font-size: 0.75rem; color: #a6adc8; }
.sym-card .warn { color: #f38ba8; }
.sym-card .exp  { color: #fab387; font-size: 0.7rem; }

/* Stat chips */
.chip {
    display: inline-block;
    background: #313244;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.8rem;
    color: #cdd6f4;
    margin: 0.2rem;
}

/* Section divider */
.section-title {
    color: #cba6f7;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid #313244;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Retrain buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <h1>🧮 Handwritten Equation Solver</h1>
    <p>Upload a photo of any handwritten math equation</p>
</div>
""", unsafe_allow_html=True)

# Load model and transform
model = load_model()
transform = get_transform()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🧠 About")
    st.markdown("""
    **Model:** LeNet-inspired CNN  
    **Symbols:** 19 classes  
    **Solver:** SymPy  
    """)

    st.markdown("---")
    st.markdown("### ✅ Supported")
    st.markdown("""
| Type | Example |
|---|---|
| Arithmetic | `3 + 5 =` |
| Linear | `3x + 5 = 11` |
| Quadratic | `x² + 5x + 6 = 0` |
| Brackets | `2(x+1) = 8` |
| Exponents | `x² + 3 = 12` |
    """)

    st.markdown("---")
    st.markdown("### ✍️ Tips")
    st.markdown("""
- White paper, black pen  
- Space out symbols clearly  
- Exponents: write **small + raised**  
- Brackets: write **tall + curved**  
- Aim for ~60px equation height  
    """)

# ── Upload strip ──
st.markdown('<div class="section-title">📤 Upload Equation Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "JPG or PNG — handwritten equation on white background",
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed'
)

# ── No upload state ──
if uploaded_file is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="step-card"><h4>Step 1</h4><p>✂️ Segment image into individual symbol crops</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="step-card"><h4>Step 2</h4><p>🧠 CNN classifies each symbol with confidence score</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="step-card"><h4>Step 3</h4><p>🔢 SymPy parses and solves the equation</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

# ── Main pipeline ──
if uploaded_file is not None:
    temp_path = Path(tempfile.gettempdir()) / f"uploaded_image_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        test_img = cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            st.error("❌ Invalid image file! Upload a valid JPG or PNG.")
        else:
            with st.spinner("🔄 Analysing equation..."):
                _, crops = segment_symbols(str(temp_path))
                if not crops:
                    st.warning("⚠️ No symbols detected. Try a clearer image with better spacing.")
                else:
                    symbols, equation_str, crops, all_predictions = recognize_equation(
                        str(temp_path), model, transform
                    )
                    if not symbols:
                        st.error("❌ Could not recognise any symbols. Check image quality.")
                    else:
                        uncertain_count = sum(1 for p in all_predictions if p.get('uncertain'))
                        avg_conf = sum(p['confidence'] for p in all_predictions if p.get('confidence')) / len(all_predictions)

                        # ── Stats strip ──
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Symbols Found",  len(crops))
                        m2.metric("Avg Confidence", f"{avg_conf:.0%}")
                        m3.metric("Uncertain",       uncertain_count)
                        m4.metric("Device",          str(device).upper())

                        st.markdown("---")

                        # ── Step 1: Segmentation ──
                        st.markdown('<div class="section-title">📸 Step 1 — Segmentation</div>', unsafe_allow_html=True)
                        col1a, col1b = st.columns(2)
                        with col1a:
                            orig_img = Image.open(temp_path)
                            st.image(orig_img, caption="Original", use_container_width=True)
                        with col1b:
                            orig, _ = preprocess_image(str(temp_path))
                            vis = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                            for (x, y, w, h, _, is_exp) in crops:
                                color = (255, 100, 0) if is_exp else (0, 200, 100)
                                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                            st.image(
                                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                caption=f"{len(crops)} symbols detected   🟢 normal   🟠 exponent",
                                use_container_width=True
                            )

                        # ── Step 2: Symbol Recognition ──
                        st.markdown('<div class="section-title">🔍 Step 2 — Symbol Recognition</div>', unsafe_allow_html=True)
                        n_cols = min(8, len(crops))
                        cols = st.columns(n_cols)
                        for idx, prediction in enumerate(all_predictions):
                            x, y, w, h, crop, is_exp = crops[idx]
                            pred    = prediction
                            sym     = pred['symbol'] if pred['symbol'] != '?' else '❓'
                            conf    = pred.get('confidence', 0)
                            unc     = pred.get('uncertain', False)
                            col_idx = idx % n_cols
                            with cols[col_idx]:
                                st.image(crop, width=90)
                                warn = "⚠️ " if unc else ""
                                exp  = " ᵉ" if is_exp else ""
                                dot  = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.60 else "🔴")
                                st.markdown(f"**{warn}{sym}{exp}**")
                                st.caption(f"{dot} {conf:.0%}")

                        # ── Step 3 rendered after correction table ──
                        st.markdown("---")

                        # ── Correction & Retraining ──
                        st.markdown('<div class="section-title">✏️ Correct & Retrain</div>', unsafe_allow_html=True)
                        st.caption("Fix any wrong predictions below, then save and retrain the model.")

                        correction_rows = []
                        for pred in all_predictions:
                            top3_str = " | ".join([f"{s}:{p:.2f}" for s, p in pred['top3'][:3]]) if pred['top3'] else ''
                            correction_rows.append({
                                'Symbol #':  pred['index'],
                                'Predicted': pred['symbol'],
                                'Confidence': f"{pred['confidence']:.0%}" if pred.get('confidence') is not None else '',
                                'Top 3':     top3_str,
                                'Corrected': pred['symbol']
                            })

                        corrected_symbols = edit_correction_table(correction_rows)
                        corrected_equation_str = ''.join(corrected_symbols)

                        if corrected_symbols != symbols:
                            st.success(f"✅ Corrected equation: **{corrected_equation_str}**")

                        rb1, rb2, rb3 = st.columns(3)
                        with rb1:
                            if st.button("💾 Save corrections", use_container_width=True):
                                saved = save_corrections_for_retraining(corrected_symbols, symbols, all_predictions)
                                if saved > 0:
                                    st.success(f"Saved {saved} correction(s)!")
                                else:
                                    st.warning("No corrections to save — predictions already correct.")
                        with rb2:
                            if st.button("📦 Export dataset", use_container_width=True):
                                count = export_corrections_for_retraining()
                                if count > 0:
                                    st.success(f"Exported {count} crops!")
                                else:
                                    st.warning("No crops to export. Save corrections first.")
                        with rb3:
                            if st.button("🔁 Retrain model", use_container_width=True):
                                with st.spinner("Retraining..."):
                                    total, avg_loss, accuracy, backup_path = retrain_model_once()
                                    if total == 0:
                                        st.warning("No corrected samples found.")
                                    else:
                                        st.success(f"Retrained on {total} samples  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.0%}")
                                        st.caption(f"Backup: {backup_path.name}")
                                        try:
                                            load_model.clear()
                                        except Exception:
                                            pass
                                        model = load_model()

                        # ── Step 3: Equation & Solution (uses corrected_symbols) ──
                        st.markdown("---")
                        st.markdown('<div class="section-title">🧠 Step 3 — Equation & Solution</div>', unsafe_allow_html=True)

                        solution, solution_text, expr_str = solve_equation(corrected_symbols)

                        st.markdown(f'<div class="eq-display">{corrected_equation_str}</div>', unsafe_allow_html=True)

                        if solution is not None:
                            st.markdown(
                                f'<div class="solution-box">'
                                f'<div class="label">Solution</div>'
                                f'<div class="eq">{corrected_equation_str}</div>'
                                f'<div class="ans">{solution_text}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(solution_text or "Could not solve. Check recognition results.")

                        st.markdown("---")

                        # ── Expanders ──
                        with st.expander("📋 Full Recognition Breakdown"):
                            breakdown_data = []
                            for pred in all_predictions:
                                top3_str = " | ".join([f"{s}: {p:.1%}" for s, p in pred['top3'][:3]]) if pred['top3'] else "—"
                                breakdown_data.append({
                                    'Symbol #':          pred['index'],
                                    'Recognized':        pred['symbol'],
                                    'Confidence':        f"{pred.get('confidence', 0):.0%}",
                                    'Top 3 Predictions': top3_str
                                })
                            st.dataframe(breakdown_data, use_container_width=True)

                        with st.expander("🔧 Debug Info"):
                            st.code(f"Parsed expression : {expr_str}\nDevice            : {device}\nModel classes     : {NUM_CLASSES}\nSymbols detected  : {len(symbols)}")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.caption("Make sure the image is clear and contains a valid equation.")