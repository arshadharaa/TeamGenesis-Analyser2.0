# ============================================================
# GENESIS — CELL 1
# CORE LIBRARIES + DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🧬 GENESIS")
print("AI-Powered Early Health Warning Research Prototype")
print("=" * 60)

# ------------------------------------------------------------
# LOAD DATASET
# On Render there is no interactive upload widget, so the
# dataset is loaded from a CSV committed to the repo instead.
# Put your dataset file next to this app.py and name it
# exactly as below (or edit the name to match).
# ------------------------------------------------------------

filename = "genesis_dataset.csv"

df = pd.read_csv(filename)

print("\n✅ DATASET LOADED")
print("-" * 60)
print("File    :", filename)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print(df.head())


# ============================================================
# GENESIS — CELL 2
# DATASET INSPECTION
# ============================================================

print("🧬 GENESIS DATASET INSPECTION")
print("=" * 60)

# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------

print("\nCOLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# ------------------------------------------------------------
# DATA TYPES
# ------------------------------------------------------------

print("\nDATA TYPES:")
print(df.dtypes)

# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ------------------------------------------------------------
# DATASET SHAPE
# ------------------------------------------------------------

print("\nDATASET SHAPE:")
print(df.shape)

# ------------------------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------------------------

print("\nSUMMARY:")
print(df.describe(include="all"))

print("\n✅ DATASET INSPECTION COMPLETE")


# ============================================================
# GENESIS — CELL 3
# FEATURE PREPARATION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("🧬 GENESIS FEATURE PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

feature_columns = [
    "bme688_gas_response_norm",
    "mq135_response_norm",
    "mq7_response_norm",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

target_column = "pattern_label"

# ------------------------------------------------------------
# INPUT + TARGET
# ------------------------------------------------------------

X = df[feature_columns].copy()
y = df[target_column].copy()

# ------------------------------------------------------------
# ENCODE PATTERN LABELS
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nFeatures:")
for feature in feature_columns:
    print("•", feature)

print("\nRecognized patterns:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} → {label}")

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ------------------------------------------------------------
# STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("✅ DATA PREPARATION COMPLETE")
print("-" * 60)

print("Total samples     :", len(df))
print("Training samples  :", len(X_train))
print("Testing samples   :", len(X_test))
print("Features          :", X_train.shape[1])

print("\nPattern distribution:")
print(y.value_counts())

print("\n🧬 Ready for Cell 4 — Model Training.")


# ============================================================
# GENESIS — CELL 4
# RANDOM FOREST MODEL TRAINING
# ============================================================

from sklearn.ensemble import RandomForestClassifier

print("🧬 GENESIS MODEL TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# TRAINING CHECK
# ------------------------------------------------------------

train_accuracy = model.score(X_train_scaled, y_train)

print("\n✅ MODEL TRAINING COMPLETE")
print("-" * 60)

print("Model              : Random Forest Classifier")
print("Number of trees    :", model.n_estimators)
print("Training samples   :", len(X_train))
print("Training accuracy  :", f"{train_accuracy * 100:.2f}%")

print("\n🧬 GENESIS model is ready for Cell 5 — Model Evaluation.")


# ============================================================
# GENESIS — CELL 5
# MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("🧬 GENESIS MODEL EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ------------------------------------------------------------
# TEST ACCURACY
# ------------------------------------------------------------

test_accuracy = accuracy_score(y_test, y_pred)

print("\n📊 TEST ACCURACY")
print("-" * 60)
print(f"{test_accuracy * 100:.2f}%")

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n📋 CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n🔲 CONFUSION MATRIX")
print("-" * 60)
print(cm)

# ------------------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.imshow(cm)

plt.title("GENESIS — Pattern Classification Confusion Matrix")
plt.xlabel("Predicted Pattern")
plt.ylabel("Actual Pattern")

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✅ MODEL EVALUATION COMPLETE")
print("🧬 Ready for Cell 6 — Feature Importance.")


# ============================================================
# GENESIS — CELL 6
# FEATURE IMPORTANCE
# ============================================================

print("🧬 GENESIS FEATURE IMPORTANCE")
print("=" * 60)

# ------------------------------------------------------------
# EXTRACT FEATURE IMPORTANCE
# ------------------------------------------------------------

importance_values = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance_values
})

# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n📊 FEATURE IMPORTANCE")
print("-" * 60)

print(feature_importance)

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title("GENESIS — Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Sensor / Environmental Feature")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("🧬 Ready for Cell 7 — GENESIS Prediction Engine.")


# ============================================================
# GENESIS — CELL 7
# PREDICTION ENGINE
# ============================================================

print("🧬 GENESIS PREDICTION ENGINE")
print("=" * 60)

def genesis_predict(
    bme688_gas,
    mq135_gas,
    mq7_gas,
    temperature,
    humidity,
    pressure
):
    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "bme688_gas_response_norm": bme688_gas,
        "mq135_response_norm": mq135_gas,
        "mq7_response_norm": mq7_gas,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "pressure_hpa": pressure
    }])

    # --------------------------------------------------------
    # APPLY THE SAME SCALER USED DURING TRAINING
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data[feature_columns])

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction_encoded = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    # --------------------------------------------------------
    # CONVERT BACK TO PATTERN NAME
    # --------------------------------------------------------

    predicted_pattern = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    # --------------------------------------------------------
    # MODEL CONFIDENCE
    # --------------------------------------------------------

    confidence = float(np.max(probabilities)) * 100

    # --------------------------------------------------------
    # PATTERN DISTRIBUTION
    # --------------------------------------------------------

    pattern_distribution = {
        label_encoder.inverse_transform([i])[0]: float(probabilities[i]) * 100
        for i in range(len(probabilities))
    }

    return {
        "pattern": predicted_pattern,
        "confidence": confidence,
        "distribution": pattern_distribution
    }


print("✅ Prediction engine created successfully.")
print()
print("Function:")
print("genesis_predict()")
print()
print("🧬 Ready for Cell 8 — Test Prediction.")


# ============================================================
# GENESIS — CELL 8
# TEST PREDICTION ENGINE
# ============================================================

print("🧬 GENESIS PREDICTION TEST")
print("=" * 60)

# ------------------------------------------------------------
# TEST INPUT
# Same style of values used by the GENESIS interface
# ------------------------------------------------------------

test_result = genesis_predict(
    bme688_gas=0.70,
    mq135_gas=0.30,
    mq7_gas=0.65,
    temperature=24.5,
    humidity=55.0,
    pressure=1008.0
)

# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

print("\n🔬 TEST SENSOR INPUT")
print("-" * 60)

print("BME688 Gas Response :", 0.70)
print("MQ-135 Response     :", 0.30)
print("MQ-7 Response       :", 0.65)
print("Temperature         :", 24.5, "°C")
print("Humidity            :", 55.0, "%")
print("Pressure            :", 1008.0, "hPa")

print("\n🧬 GENESIS RESULT")
print("-" * 60)

print("Predicted Pattern   :", test_result["pattern"])
print("Model Confidence    :", f"{test_result['confidence']:.2f}%")

print("\n📊 PATTERN DISTRIBUTION")
print("-" * 60)

for pattern, percentage in test_result["distribution"].items():
    print(f"{pattern:<12} : {percentage:.2f}%")

print("\n" + "=" * 60)
print("✅ PREDICTION ENGINE TEST COMPLETE")
print("🧬 Backend is ready to connect to the GENESIS website.")


# ============================================================
# GENESIS — CELL 9
# WEBSITE ANALYSIS ENGINE
# ============================================================

print("🧬 GENESIS WEBSITE ANALYSIS ENGINE")
print("=" * 60)

# ------------------------------------------------------------
# STORE LATEST ANALYSIS
# ------------------------------------------------------------

latest_analysis = {}

# ------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ------------------------------------------------------------

def run_analysis(
    bme688_gas,
    mq135_gas,
    mq7_gas,
    temperature,
    humidity,
    pressure
):
    global latest_analysis

    # Get prediction from trained GENESIS model
    result = genesis_predict(
        bme688_gas=bme688_gas,
        mq135_gas=mq135_gas,
        mq7_gas=mq7_gas,
        temperature=temperature,
        humidity=humidity,
        pressure=pressure
    )

    # --------------------------------------------------------
    # STORE COMPLETE ANALYSIS
    # --------------------------------------------------------

    latest_analysis = {
        "pattern": result["pattern"],
        "confidence": result["confidence"],
        "distribution": result["distribution"],

        "sensor_profile": {
            "BME688 Gas Response": bme688_gas,
            "MQ-135 Response": mq135_gas,
            "MQ-7 Response": mq7_gas,
            "Temperature": temperature,
            "Humidity": humidity,
            "Pressure": pressure
        }
    }

    return latest_analysis


# ------------------------------------------------------------
# TEST WEBSITE ANALYSIS FLOW
# ------------------------------------------------------------

website_test = run_analysis(
    bme688_gas=0.70,
    mq135_gas=0.30,
    mq7_gas=0.65,
    temperature=24.5,
    humidity=55.0,
    pressure=1008.0
)

# ------------------------------------------------------------
# DISPLAY TEST RESULT
# ------------------------------------------------------------

print("\n✅ WEBSITE ANALYSIS ENGINE READY")
print("-" * 60)

print("Pattern    :", website_test["pattern"])
print("Confidence:", f"{website_test['confidence']:.2f}%")

print("\nSensor profile stored:")
for name, value in website_test["sensor_profile"].items():
    print(f" • {name}: {value}")

print("\n" + "=" * 60)
print("🧬 GENESIS backend is ready for website integration.")



# ============================================================
# GENESIS — CELL 10
# FINAL PREMIUM MULTI-PAGE RESEARCH INTERFACE
# (VISUAL POLISH PASS — LOGIC UNCHANGED)
#
# IMPORTANT:
# 1. Run CELL 8
# 2. Run CELL 9
# 3. Run THIS CELL 10
#
# Uses:
# - genesis_css from Cell 9
# - genesis_predict() from the previous model cell
# - Premium animated HTML graphs
# - Correctly passes analysis values to ALL visual pages
#
# WHAT CHANGED IN THIS PASS:
# Only the CSS (premium_css block + the inline <style> in the
# Home hero) was touched, to make every page share the same
# spacing scale, alignment, card treatment, and typography.
# No Python function, gr.* component, event binding, or model
# call was modified. All ids/classes referenced by callbacks
# (prediction_output, probability_output, bar_plot, pie_plot,
# profile_plot, etc.) are untouched.
# ============================================================

import gradio as gr
import numpy as np
import pandas as pd  # ensures pd is available even if Cell 1 wasn't re-run in this kernel


# ============================================================
# PAGE CONTROLLER
# ============================================================

def show_page(page_name):

    names = [
        "home",
        "analysis",
        "result",
        "bar",
        "pie",
        "profile"
    ]

    return [
        gr.update(visible=(page_name == name))
        for name in names
    ]


# ============================================================
# CLEAN PATTERN NAMES
# ============================================================

def clean_pattern_name(name):

    if name is None:
        return "Unknown"

    name = str(name).replace("_", " ").strip()

    if name.lower() == "unknown":
        return "Unknown"

    if name.lower().startswith("pattern"):

        parts = name.split()

        if len(parts) > 1:
            return "Pattern " + parts[-1].upper()

    return name.title()


# ============================================================
# PREMIUM VISUAL CSS
# ============================================================

premium_css = """

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* ==========================================================
   DESIGN TOKENS — "Signal Bench"
   An instrument-panel language for a sensor-reading system:
   hairline panels, bracket corners, a brass dial accent, and
   monospace numerals for anything that is a measured value.
   ========================================================== */

:root {

    --g-space-1: 6px;
    --g-space-2: 10px;
    --g-space-3: 16px;
    --g-space-4: 24px;
    --g-space-5: 36px;
    --g-space-6: 52px;

    --g-radius-sm: 6px;
    --g-radius-md: 8px;
    --g-radius-lg: 10px;

    --g-ink: #edf1ee;
    --g-ink-dim: #8d968f;
    --g-ink-faint: #5b625b;

    --g-amber: #f2a93b;
    --g-amber-light: #f7c26e;
    --g-amber-deep: #a86a1c;

    --g-teal: #5fd6c4;

    --g-border: rgba(242,169,59,0.20);
    --g-border-soft: rgba(255,255,255,0.07);
    --g-hairline: rgba(255,255,255,0.09);

    --g-surface: linear-gradient(
        175deg,
        rgba(19,23,20,0.97),
        rgba(9,12,10,0.99)
    );

    --g-shadow-lg:
        0 24px 60px rgba(0,0,0,0.40);

    --g-shadow-sm:
        0 10px 24px rgba(0,0,0,0.30);

    --g-content-width: 1080px;

    --g-font-display: 'Space Grotesk', 'Segoe UI', sans-serif;
    --g-font-body: 'Inter', 'Segoe UI', sans-serif;
    --g-font-mono: 'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace;

}


/* ==========================================================
   GLOBAL
   ========================================================== */

body {

    background:
        radial-gradient(
            ellipse at 12% 0%,
            rgba(242,169,59,0.055),
            transparent 32%
        ),
        repeating-linear-gradient(
            0deg,
            rgba(255,255,255,0.012) 0px,
            rgba(255,255,255,0.012) 1px,
            transparent 1px,
            transparent 34px
        ),
        repeating-linear-gradient(
            90deg,
            rgba(255,255,255,0.012) 0px,
            rgba(255,255,255,0.012) 1px,
            transparent 1px,
            transparent 34px
        ),
        #0a0e0c !important;

    font-family:
        var(--g-font-body) !important;

}

* {
    box-sizing: border-box;
}


/* ==========================================================
   NAVIGATION
   ========================================================== */

.nav {

    position: sticky !important;

    top: 0;

    z-index: 100;

    display: flex !important;

    align-items: center !important;

    justify-content: center !important;

    gap: 2px !important;

    padding: 13px 18px !important;

    margin-bottom: 20px !important;

    background:
        rgba(10,14,12,0.95) !important;

    backdrop-filter:
        blur(16px);

    border-bottom:
        1px solid var(--g-hairline) !important;

}


/* ==========================================================
   NAVIGATION BUTTONS
   ========================================================== */

.nav-btn {

    position: relative !important;

    background:
        transparent !important;

    border:
        1px solid transparent !important;

    color:
        #868d87 !important;

    font-family:
        var(--g-font-body) !important;

    font-size:
        12px !important;

    font-weight:
        500 !important;

    letter-spacing:
        0.2px !important;

    padding:
        7px 15px !important;

    border-radius:
        var(--g-radius-sm) !important;

    transition:
        all 0.2s ease !important;

}


.nav-btn:hover {

    color:
        var(--g-amber-light) !important;

    background:
        rgba(242,169,59,0.06) !important;

    border-color:
        var(--g-border) !important;

}


/* ==========================================================
   NAV BRAND
   ========================================================== */

.nav-brand {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

    margin-right:
        14px;

    padding-right:
        18px;

    border-right:
        1px solid var(--g-hairline);

}


.nav-brand-mark {

    width:
        30px;

    height:
        30px;

    border:
        1px solid var(--g-amber);

    border-radius:
        var(--g-radius-sm);

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    color:
        var(--g-amber-light);

    font-family:
        var(--g-font-mono);

    font-size:
        15px;

    font-weight:
        600;

    background:
        rgba(242,169,59,0.06);

    transition:
        all 0.25s ease;

}


.nav-brand-mark:hover {

    background:
        rgba(242,169,59,0.13);

    border-color:
        var(--g-amber-light);

}


.nav-brand-text {

    color:
        var(--g-ink);

    font-family:
        var(--g-font-display);

    font-size:
        13px;

    font-weight:
        600;

    letter-spacing:
        1px;

}


/* ==========================================================
   PAGE — shared vertical rhythm + centered content column
   ========================================================== */

.page {

    min-height:
        76vh;

    padding:
        var(--g-space-6) var(--g-space-4) 64px;

    animation:
        pageEnter 0.5s ease both;

}

.page > * {

    max-width:
        var(--g-content-width);

    margin-left:
        auto;

    margin-right:
        auto;

}


@keyframes pageEnter {

    from {

        opacity:
            0;

        transform:
            translateY(10px);

    }

    to {

        opacity:
            1;

        transform:
            translateY(0);

    }

}


/* ==========================================================
   VISUAL HEADERS
   Left-anchored, panel-style rather than centered marketing
   copy — reads like a section header on an instrument screen.
   ========================================================== */

.visual-title {

    text-align:
        left;

    font-family:
        var(--g-font-display);

    font-size:
        clamp(24px, 3vw, 32px);

    font-weight:
        600;

    letter-spacing:
        0.2px;

    color:
        var(--g-ink);

    margin:
        0 0 8px;

}


.visual-description {

    text-align:
        left;

    max-width:
        620px;

    margin:
        0 0 var(--g-space-5);

    color:
        var(--g-ink-dim);

    font-size:
        13.5px;

    line-height:
        1.7;

}


/* ==========================================================
   BADGE — bracket tag, instrument-panel language instead
   of a tracked-out all-caps pill.
   ========================================================== */

.page-badge {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        6px;

    padding:
        4px 0;

    margin-bottom:
        var(--g-space-2);

    color:
        var(--g-amber-light);

    font-family:
        var(--g-font-mono);

    font-size:
        11px;

    font-weight:
        500;

    letter-spacing:
        0.4px;

}

.page-badge::before {

    content:
        "";

    width:
        14px;

    height:
        1px;

    background:
        var(--g-amber);

}


/* ==========================================================
   PREMIUM PANEL — hairline bracket-corner treatment shared
   by graph cards, result card, and analysis panels.
   ========================================================== */

.graph-card {

    position:
        relative;

    max-width:
        var(--g-content-width);

    margin:
        var(--g-space-5) auto;

    padding:
        36px 40px;

    background:
        var(--g-surface);

    border:
        1px solid var(--g-hairline);

    box-shadow:
        var(--g-shadow-lg);

}


.graph-card::before,
.graph-card::after {

    content:
        "";

    position:
        absolute;

    width:
        20px;

    height:
        20px;

    border:
        1px solid var(--g-amber);

    opacity:
        0.55;

    pointer-events:
        none;

}


.graph-card::before {

    top:
        -1px;

    left:
        -1px;

    border-right:
        none;

    border-bottom:
        none;

}


.graph-card::after {

    bottom:
        -1px;

    right:
        -1px;

    border-left:
        none;

    border-top:
        none;

}


/* ==========================================================
   GRAPH HEADER
   ========================================================== */

.graph-heading {

    font-family:
        var(--g-font-display);

    font-size:
        15px;

    font-weight:
        600;

    color:
        var(--g-ink);

    margin-bottom:
        4px;

}


.graph-subheading {

    color:
        var(--g-ink-faint);

    font-size:
        11.5px;

    margin-bottom:
        var(--g-space-5);

}


/* ==========================================================
   BAR GRAPH — styled as a waveform / signal readout
   ========================================================== */

.bar-chart {

    display:
        flex;

    align-items:
        flex-end;

    justify-content:
        space-evenly;

    gap:
        30px;

    height:
        340px;

    padding:
        var(--g-space-4) var(--g-space-3) 0;

    border-bottom:
        1px solid var(--g-hairline);

    position:
        relative;

}

.bar-chart::before {

    content:
        "";

    position:
        absolute;

    left:
        0;

    right:
        0;

    top:
        25%;

    height:
        1px;

    background:
        repeating-linear-gradient(
            90deg,
            rgba(255,255,255,0.07) 0,
            rgba(255,255,255,0.07) 4px,
            transparent 4px,
            transparent 10px
        );

}

.bar-chart::after {

    content:
        "";

    position:
        absolute;

    left:
        0;

    right:
        0;

    top:
        62.5%;

    height:
        1px;

    background:
        repeating-linear-gradient(
            90deg,
            rgba(255,255,255,0.07) 0,
            rgba(255,255,255,0.07) 4px,
            transparent 4px,
            transparent 10px
        );

}


.bar-column {

    height:
        100%;

    flex:
        1;

    max-width:
        120px;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        flex-end;

    align-items:
        center;

    position:
        relative;

    z-index:
        1;

}


.bar-value {

    color:
        var(--g-amber-light);

    font-family:
        var(--g-font-mono);

    font-size:
        12px;

    font-weight:
        600;

    margin-bottom:
        8px;

}


.bar {

    width:
        44px;

    max-width:
        100%;

    min-height:
        4px;

    background:
        linear-gradient(
            180deg,
            var(--g-amber-light),
            var(--g-amber-deep)
        );

    transform-origin:
        bottom;

    animation:
        barRise 0.85s cubic-bezier(.2,.8,.2,1) both;

}


@keyframes barRise {

    from {

        transform:
            scaleY(0);

        opacity:
            0;

    }

    to {

        transform:
            scaleY(1);

        opacity:
            1;

    }

}


.bar-label {

    margin-top:
        12px;

    color:
        var(--g-ink-dim);

    font-family:
        var(--g-font-mono);

    font-size:
        10.5px;

    font-weight:
        500;

    text-align:
        center;

}


/* ==========================================================
   PIE GRAPH — kept as a dial-style ring rather than a
   default donut chart, to match the instrument language.
   ========================================================== */

.pie-layout {

    display:
        flex;

    align-items:
        center;

    justify-content:
        flex-start;

    gap:
        64px;

    flex-wrap:
        wrap;

}


.pie-chart {

    width:
        270px;

    height:
        270px;

    border-radius:
        50%;

    position:
        relative;

    animation:
        piePop 0.7s cubic-bezier(.2,.8,.2,1) both;

    flex-shrink:
        0;

    border:
        1px solid var(--g-hairline);

    padding:
        10px;

}


@keyframes piePop {

    from {

        opacity:
            0;

        transform:
            scale(0.7);

    }

    to {

        opacity:
            1;

        transform:
            scale(1);

    }

}


.pie-hole {

    position:
        absolute;

    width:
        108px;

    height:
        108px;

    top:
        50%;

    left:
        50%;

    transform:
        translate(-50%,-50%);

    border-radius:
        50%;

    background:
        #0d100e;

    border:
        1px solid var(--g-hairline);

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    text-align:
        center;

    color:
        var(--g-ink-dim);

    font-family:
        var(--g-font-mono);

    font-size:
        9.5px;

    letter-spacing:
        0.5px;

    line-height:
        1.5;

}


.legend {

    min-width:
        230px;

}


.legend-item {

    display:
        flex;

    align-items:
        center;

    gap:
        11px;

    padding:
        10px 0;

    border-bottom:
        1px solid var(--g-hairline);

    color:
        var(--g-ink-dim);

    font-size:
        12.5px;

}

.legend-item:last-child {

    border-bottom:
        none;

}


.legend-dot {

    width:
        8px;

    height:
        8px;

    flex-shrink:
        0;

}


.legend-name {

    flex:
        1;

}


.legend-value {

    color:
        var(--g-amber-light);

    font-family:
        var(--g-font-mono);

    font-weight:
        600;

}


/* ==========================================================
   SENSOR PROFILE — instrument readout tiles
   ========================================================== */

.sensor-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2,1fr);

    gap:
        1px;

    background:
        var(--g-hairline);

    border:
        1px solid var(--g-hairline);

}


.sensor-item {

    padding:
        20px 22px;

    background:
        #0d110f;

    transition:
        background 0.2s ease;

}


.sensor-item:hover {

    background:
        #111512;

}


.sensor-name {

    color:
        var(--g-ink-dim);

    font-size:
        11px;

    margin-bottom:
        9px;

}


.sensor-value {

    color:
        var(--g-ink);

    font-family:
        var(--g-font-mono);

    font-size:
        22px;

    font-weight:
        600;

}


.sensor-track {

    height:
        3px;

    margin-top:
        13px;

    background:
        #1c211d;

    overflow:
        hidden;

}


.sensor-fill {

    height:
        100%;

    background:
        var(--g-amber);

    transform-origin:
        left;

    animation:
        sensorGrow 1s ease both;

}


@keyframes sensorGrow {

    from {

        transform:
            scaleX(0);

    }

    to {

        transform:
            scaleX(1);

    }

}


/* ==========================================================
   SHARED SECTION TITLES
   ========================================================== */

.section-title {

    text-align:
        left;

    font-family:
        var(--g-font-display);

    font-size:
        21px;

    font-weight:
        600;

    letter-spacing:
        0.2px;

    color:
        var(--g-ink);

    margin-bottom:
        8px;

}


.section-description {

    text-align:
        left;

    max-width:
        620px;

    color:
        var(--g-ink-dim);

    font-size:
        13px;

    line-height:
        1.7;

}


/* ==========================================================
   INFO CARDS (Home — "Why GENESIS" row)
   No decorative numbering: this content is three parallel
   ideas, not a sequence.
   ========================================================== */

.info-card {

    height:
        100%;

    padding:
        24px 22px;

    background:
        rgba(255,255,255,0.015);

    border:
        1px solid var(--g-hairline);
    border-left:
        2px solid var(--g-amber-deep);

    transition:
        border-color 0.2s ease, background 0.2s ease;

}


.info-card:hover {

    border-left-color:
        var(--g-amber-light);

    background:
        rgba(255,255,255,0.028);

}


.info-heading {

    font-family:
        var(--g-font-display);

    color:
        var(--g-ink);

    font-size:
        14.5px;

    font-weight:
        600;

    margin-bottom:
        8px;

}


.info-text {

    color:
        var(--g-ink-dim);

    font-size:
        12.5px;

    line-height:
        1.7;

}


/* ==========================================================
   QUERY / ASK GENESIS BOX
   ========================================================== */

.query-box {

    text-align:
        left;

    max-width:
        720px;

    margin:
        var(--g-space-6) 0 var(--g-space-4);

}


.query-answer {

    max-width:
        var(--g-content-width);

    margin:
        var(--g-space-3) 0 0;

    padding:
        18px 22px;

    background:
        rgba(255,255,255,0.015);

    border-left:
        2px solid var(--g-amber);

    color:
        #c3bfb6;

    font-size:
        13px;

    line-height:
        1.75;

}


/* ==========================================================
   RESULT PAGE
   ========================================================== */

.result-card {

    max-width:
        var(--g-content-width);

    margin:
        0 auto;

    padding:
        34px 38px;

    background:
        var(--g-surface);

    border:
        1px solid var(--g-hairline);

    box-shadow:
        var(--g-shadow-lg);

    text-align:
        left;

}


.result-label {

    color:
        var(--g-ink-dim);

    font-family:
        var(--g-font-mono);

    font-size:
        10.5px;

    letter-spacing:
        0.5px;

    margin-bottom:
        var(--g-space-3);

}


.pattern-result {

    font-family:
        var(--g-font-display);

    color:
        var(--g-ink);

    font-size:
        clamp(30px, 4vw, 42px);

    font-weight:
        600;

}


.confidence {

    margin-top:
        var(--g-space-3);

    color:
        var(--g-ink-dim);

    font-size:
        12px;

}


.confidence strong {

    display:
        block;

    color:
        var(--g-amber-light);

    font-family:
        var(--g-font-mono);

    font-size:
        22px;

    margin-top:
        4px;

}


.status {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        7px;

    margin-top:
        var(--g-space-3);

    color:
        var(--g-teal);

    font-family:
        var(--g-font-mono);

    font-size:
        10.5px;

    letter-spacing:
        0.4px;

}

.status::before {

    content:
        "";

    width:
        6px;

    height:
        6px;

    border-radius:
        50%;

    background:
        var(--g-teal);

    box-shadow:
        0 0 8px var(--g-teal);

}


.probability-row {

    max-width:
        var(--g-content-width);

    margin:
        0 auto var(--g-space-3);

}


.probability-label {

    display:
        flex;

    justify-content:
        space-between;

    color:
        var(--g-ink-dim);

    font-family:
        var(--g-font-mono);

    font-size:
        11.5px;

    margin-bottom:
        6px;

}


.probability-track {

    height:
        6px;

    background:
        #171b18;

    overflow:
        hidden;

    border:
        1px solid var(--g-hairline);

}


.probability-fill {

    height:
        100%;

    background:
        linear-gradient(
            90deg,
            var(--g-amber-deep),
            var(--g-amber-light)
        );

    transition:
        width 0.6s cubic-bezier(.2,.8,.2,1);

}


/* ==========================================================
   ANALYSIS PAGE — input / pipeline panels
   ========================================================== */

.panel {

    height:
        100%;

    padding:
        28px;

    background:
        var(--g-surface);

    border:
        1px solid var(--g-hairline);

    box-shadow:
        var(--g-shadow-sm);

}


.pipeline-flow {

    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        0;

    margin-top:
        var(--g-space-4);

    border:
        1px solid var(--g-hairline);

}


.pipeline-node {

    padding:
        11px 14px;

    background:
        rgba(255,255,255,0.015);

    border-right:
        1px solid var(--g-hairline);

    color:
        #cfc9be;

    font-family:
        var(--g-font-mono);

    font-size:
        11px;

    flex:
        1;

    text-align:
        center;

}

.pipeline-node:last-child {

    border-right:
        none;

    color:
        var(--g-amber-light);

}


.pipeline-arrow {

    display:
        none;

}


.analyze-btn {

    margin-top:
        var(--g-space-3) !important;

}


/* ==========================================================
   DISCLAIMER
   ========================================================== */

.disclaimer {

    max-width:
        var(--g-content-width);

    margin:
        var(--g-space-5) auto 0;

    padding:
        16px 20px;

    border-top:
        1px solid var(--g-hairline);

    color:
        var(--g-ink-faint);

    font-size:
        11.5px;

    line-height:
        1.7;

    text-align:
        left;

}


/* ==========================================================
   MOBILE
   ========================================================== */

@media(max-width:700px) {

    .nav {

        overflow-x:
            auto;

        justify-content:
            flex-start !important;

    }

    .nav-brand {

        flex-shrink:
            0;

    }

    .page {

        padding:
            var(--g-space-5) var(--g-space-3) 44px;

    }

    .graph-card {

        padding:
            26px 20px;

    }

    .bar-chart {

        gap:
            10px;

        padding:
            var(--g-space-3) 5px 0;

    }

    .bar {

        width:
            30px;

    }

    .sensor-grid {

        grid-template-columns:
            1fr;

    }

    .pie-layout {

        gap:
            var(--g-space-4);

        justify-content:
            center;

    }

    .visual-title,
    .visual-description,
    .section-title,
    .section-description,
    .query-box,
    .result-card {

        text-align:
            center;

    }

}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {

    text-align:
        center;

    padding:
        44px 20px;

    margin-top:
        var(--g-space-4);

    border-top:
        1px solid var(--g-hairline);

    color:
        var(--g-ink-faint);

    font-family:
        var(--g-font-mono);

    font-size:
        10.5px;

    line-height:
        1.9;

}

"""



# ============================================================
# GRAPH GENERATORS
# ============================================================

def create_bar_html(probabilities):

    labels = [
        "Pattern A",
        "Pattern B",
        "Pattern C",
        "Pattern D",
        "Unknown"
    ]

    keys = [
        "Pattern_A",
        "Pattern_B",
        "Pattern_C",
        "Pattern_D",
        "Unknown"
    ]

    values = []

    for key in keys:

        value = probabilities.get(key, 0)

        try:
            value = float(value)
        except:
            value = 0.0

        values.append(value)


    max_value = max(values) if max(values) > 0 else 100


    bars = ""

    for i, (label, value) in enumerate(
        zip(labels, values)
    ):

        # Keep bars proportional to the highest value.
        height = max(
            2,
            (value / max_value) * 100
        )


        bars += f"""

        <div class="bar-column">

            <div class="bar-value">
                {value:.1f}%
            </div>

            <div
                class="bar"
                style="
                    height:{height}%;
                    animation-delay:{i * 0.10}s;
                "
            ></div>

            <div class="bar-label">
                {label}
            </div>

        </div>

        """


    return f"""

    <div class="graph-card">

        <div class="graph-heading">
            Model Probability Distribution
        </div>

        <div class="graph-subheading">
            Relative classification probability for the current sample
        </div>

        <div class="bar-chart">
            {bars}
        </div>

    </div>

    """


# ============================================================
# PIE GENERATOR
# ============================================================

def create_pie_html(probabilities):

    labels = [
        ("Pattern A", "Pattern_A"),
        ("Pattern B", "Pattern_B"),
        ("Pattern C", "Pattern_C"),
        ("Pattern D", "Pattern_D"),
        ("Unknown", "Unknown")
    ]


    values = []

    for _, key in labels:

        try:
            values.append(
                float(probabilities.get(key, 0))
            )
        except:
            values.append(0.0)


    total = sum(values)


    if total <= 0:

        values = [
            20,
            20,
            20,
            20,
            20
        ]

        total = 100


    percentages = [
        (v / total) * 100
        for v in values
    ]


    colors = [
        "#d49a69",
        "#b87947",
        "#9b5f35",
        "#754325",
        "#4d4a46"
    ]


    current = 0

    stops = []


    for pct, color in zip(
        percentages,
        colors
    ):

        end = current + pct

        stops.append(
            f"{color} {current:.2f}% {end:.2f}%"
        )

        current = end


    gradient = ", ".join(stops)


    legend = ""


    for (
        (label, _),
        pct,
        color
    ) in zip(
        labels,
        percentages,
        colors
    ):

        legend += f"""

        <div class="legend-item">

            <div
                class="legend-dot"
                style="background:{color};"
            ></div>

            <div class="legend-name">
                {label}
            </div>

            <div class="legend-value">
                {pct:.1f}%
            </div>

        </div>

        """


    return f"""

    <div class="graph-card">

        <div class="graph-heading">
            Pattern Composition
        </div>

        <div class="graph-subheading">
            Proportional distribution of model probabilities
        </div>

        <div class="pie-layout">

            <div
                class="pie-chart"
                style="
                    background:
                    conic-gradient(
                        {gradient}
                    );
                "
            >

                <div class="pie-hole">
                    GENESIS<br>
                    RESPONSE
                </div>

            </div>

            <div class="legend">
                {legend}
            </div>

        </div>

    </div>

    """


# ============================================================
# SENSOR PROFILE GENERATOR
# ============================================================

def create_profile_html(
    bme,
    mq135,
    mq7,
    temp,
    humidity,
    pressure
):

    sensors = [

        (
            "BME688 Gas Response",
            bme,
            1.0
        ),

        (
            "MQ-135 Response",
            mq135,
            1.0
        ),

        (
            "MQ-7 Response",
            mq7,
            1.0
        ),

        (
            "Temperature",
            temp,
            40
        ),

        (
            "Humidity",
            humidity,
            100
        ),

        (
            "Pressure",
            pressure,
            1100
        )

    ]


    cards = ""


    for i, (
        name,
        value,
        maximum
    ) in enumerate(sensors):

        try:

            value = float(value)

        except:

            value = 0.0


        try:

            maximum = float(maximum)

        except:

            maximum = 1.0


        percentage = min(
            100,
            max(
                3,
                (value / maximum) * 100
            )
        )


        cards += f"""

        <div
            class="sensor-item"
            style="
                animation:
                pageEnter 0.5s ease {i*0.08}s both;
            "
        >

            <div class="sensor-name">
                {name}
            </div>

            <div class="sensor-value">
                {value:.2f}
            </div>

            <div class="sensor-track">

                <div
                    class="sensor-fill"
                    style="
                        width:{percentage}%;
                        animation-delay:{i*0.08}s;
                    "
                ></div>

            </div>

        </div>

        """


    return f"""

    <div class="graph-card">

        <div class="graph-heading">
            Current Sensor Response
        </div>

        <div class="graph-subheading">
            Six-dimensional response fingerprint of the submitted sample
        </div>

        <div class="sensor-grid">

            {cards}

        </div>

    </div>

    """


# ============================================================
# QUERY ANSWER
# ============================================================

def answer_query(question):

    if not question or not question.strip():

        return """

        <div class="query-answer">

            Enter a question to explore GENESIS.

        </div>

        """


    q = question.lower().strip()


    if "genesis" in q:

        answer = """

        <b>GENESIS</b> is a machine-learning research prototype
        that studies combined responses from multiple sensors
        and identifies patterns within a controlled research dataset.

        """


    elif "sensor" in q:

        answer = """

        The prototype combines BME688, MQ-135 and MQ-7 response
        values with temperature, humidity and pressure to form
        a multi-dimensional sensor-response profile.

        """


    elif (
        "cancer" in q
        or "illness" in q
        or "disease" in q
    ):

        answer = """

        GENESIS explores the concept of identifying unusual
        response patterns that may deserve further investigation.
        A classification result is <b>not proof of illness</b>
        and the prototype is not a diagnostic system.

        """


    elif (
        "percentage" in q
        or "confidence" in q
    ):

        answer = """

        A higher percentage means the model assigned greater
        probability to that learned pattern compared with the
        other classes. It is <b>not</b> a percentage chance of
        having a disease.

        """


    elif "unknown" in q:

        answer = """

        <b>Unknown</b> represents a response that does not closely
        resemble the learned pattern classes. This helps the
        system avoid forcing every response into a familiar class.

        """


    elif (
        "ai" in q
        or "machine" in q
        or "model" in q
    ):

        answer = """

        The machine-learning model learns relationships between
        the six measurements and the pattern labels in the
        research dataset. A new sample is then compared against
        those learned relationships.

        """


    else:

        answer = """

        GENESIS studies combinations of measurements rather than
        relying on a single sensor value. The resulting response
        profile is evaluated by the trained machine-learning model.

        """


    return f"""

    <div class="query-answer">

        {answer}

    </div>

    """


# ============================================================
# APPLICATION
# ============================================================

# Use the base stylesheet already available in the notebook.
# Cell 10 no longer crashes if the variable is named CSS instead
# of genesis_css. This does not alter any page content.
base_css = globals().get("genesis_css", globals().get("CSS", ""))


with gr.Blocks(

    title="GENESIS | Research Interface",

    css=base_css + premium_css

) as demo:


    # NAVIGATION
    # ========================================================

    with gr.Row(
        elem_classes="nav"
    ):

        gr.HTML("""

        <div class="nav-brand">

            <div class="nav-brand-mark">
                G
            </div>

            <div class="nav-brand-text">
                GENESIS
            </div>

        </div>

        """)


        home_btn = gr.Button(
            "Home",
            elem_classes="nav-btn"
        )


        analysis_btn = gr.Button(
            "Analysis",
            elem_classes="nav-btn"
        )


        result_btn = gr.Button(
            "Result",
            elem_classes="nav-btn"
        )


        bar_btn = gr.Button(
            "Bar Graph",
            elem_classes="nav-btn"
        )


        pie_btn = gr.Button(
            "Pie Graph",
            elem_classes="nav-btn"
        )


        profile_btn = gr.Button(
            "Sensor Profile",
            elem_classes="nav-btn"
        )


    # ========================================================
    # HOME
    # ========================================================

    with gr.Column(

        visible=True,

        elem_classes="page"

    ) as home_page:


        # ====================================================
        # PREMIUM HOME HERO — HOME PAGE ONLY
        # ====================================================

        gr.HTML("""
        <style>

        .genesis-home-wrap {
            width: 100%;
            max-width: 1180px;
            margin: 0 auto;
        }

        .genesis-home-hero {
            position: relative;
            min-height: 66vh;
            padding: 52px 52px 46px;
            display: grid;
            grid-template-columns: 1.05fr .95fr;
            align-items: center;
            gap: 46px;
            overflow: hidden;
            border: 1px solid rgba(196,122,60,.18);
            border-radius: 26px;
            background:
                radial-gradient(circle at 82% 48%, rgba(196,122,60,.11), transparent 30%),
                radial-gradient(circle at 12% 18%, rgba(88,183,154,.06), transparent 28%),
                linear-gradient(145deg, rgba(21,22,21,.98), rgba(8,9,9,.99));
            box-shadow:
                0 30px 80px rgba(0,0,0,.32),
                inset 0 1px rgba(255,255,255,.04);
        }

        .genesis-home-hero::before {
            content: "";
            position: absolute;
            width: 520px;
            height: 520px;
            right: -205px;
            top: -190px;
            border-radius: 50%;
            border: 1px solid rgba(196,122,60,.09);
            box-shadow:
                0 0 0 36px rgba(196,122,60,.02),
                0 0 0 78px rgba(196,122,60,.011);
            pointer-events: none;
        }

        .genesis-home-copy {
            position: relative;
            z-index: 2;
        }

        .genesis-home-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 13px;
            margin-bottom: 22px;
            border-radius: 30px;
            background: rgba(88,183,154,.05);
            border: 1px solid rgba(88,183,154,.16);
            color: #8ccbb5;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 1.8px;
        }

        .genesis-home-status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #63d6a8;
            box-shadow:
                0 0 6px #63d6a8,
                0 0 15px rgba(99,214,168,.55);
            animation: genesisStatusBlink 1.4s ease-in-out infinite;
        }

        @keyframes genesisStatusBlink {
            0%,100% { opacity: 1; transform: scale(1); }
            50% { opacity: .32; transform: scale(.72); }
        }

        .genesis-home-eyebrow {
            color: #a9a39b;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 3px;
            margin-bottom: 13px;
        }

        .genesis-home-title {
            margin: 0;
            color: #f1eee9;
            font-size: clamp(58px, 6.6vw, 92px);
            font-weight: 800;
            letter-spacing: 7px;
            line-height: .95;
            text-shadow: 0 0 35px rgba(196,122,60,.09);
        }

        .genesis-home-line {
            width: 72px;
            height: 2px;
            margin: 25px 0 20px;
            background: linear-gradient(90deg, #c47a3c, transparent);
        }

        .genesis-home-tagline {
            color: #e0a064;
            font-size: 17px;
            font-weight: 600;
            letter-spacing: 1px;
            margin-bottom: 18px;
        }

        .genesis-home-description {
            max-width: 570px;
            color: #aaa49c;
            font-size: 13px;
            line-height: 1.85;
        }

        .genesis-home-description strong {
            color: #ded7ce;
            font-weight: 650;
        }

        .genesis-home-data {
            display: grid;
            grid-template-columns: repeat(3,minmax(0,1fr));
            gap: 11px;
            max-width: 610px;
            margin-top: 28px;
        }

        .genesis-home-data-cell {
            padding: 13px 14px;
            border-radius: 12px;
            background: rgba(255,255,255,.02);
            border: 1px solid rgba(255,255,255,.06);
            transition: transform .25s ease, border-color .25s ease, background .25s ease;
        }

        .genesis-home-data-cell:hover {
            transform: translateY(-3px);
            border-color: rgba(196,122,60,.24);
            background: rgba(196,122,60,.04);
        }

        .genesis-home-data-value {
            display: block;
            color: #d49a69;
            font-size: 11px;
            font-weight: 750;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }

        .genesis-home-data-label {
            color: #77736e;
            font-size: 9px;
            line-height: 1.4;
        }

        .genesis-sensor-field {
            position: relative;
            width: min(420px,100%);
            aspect-ratio: 1 / 1;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .genesis-orbit {
            position: absolute;
            width: 72%;
            height: 72%;
            border-radius: 50%;
            border: 1px solid rgba(196,122,60,.18);
            animation: genesisOrbit 18s linear infinite;
        }

        .genesis-orbit::before {
            content: "";
            position: absolute;
            width: 7px;
            height: 7px;
            top: 5px;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 50%;
            background: #c47a3c;
            box-shadow: 0 0 10px rgba(196,122,60,.78);
        }

        .genesis-orbit-two {
            width: 88%;
            height: 88%;
            border-color: rgba(88,183,154,.10);
            animation: genesisOrbitReverse 25s linear infinite;
        }

        .genesis-orbit-two::before {
            top: auto;
            bottom: 4px;
            background: #63d6a8;
            box-shadow: 0 0 10px rgba(99,214,168,.72);
        }

        @keyframes genesisOrbit {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes genesisOrbitReverse {
            from { transform: rotate(360deg); }
            to { transform: rotate(0deg); }
        }

        .genesis-home-core {
            position: relative;
            width: 145px;
            height: 145px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #e0a064;
            font-size: 62px;
            font-weight: 850;
            letter-spacing: -2px;
            background: radial-gradient(circle, rgba(196,122,60,.13), rgba(10,11,11,.98) 68%);
            border: 1px solid rgba(196,122,60,.36);
            box-shadow:
                0 0 42px rgba(196,122,60,.12),
                inset 0 0 35px rgba(196,122,60,.04);
        }

        .genesis-home-core::after {
            content: "";
            position: absolute;
            width: 9px;
            height: 9px;
            top: 18px;
            right: 20px;
            border-radius: 50%;
            background: #63d6a8;
            box-shadow:
                0 0 5px #63d6a8,
                0 0 14px rgba(99,214,168,.85),
                0 0 28px rgba(99,214,168,.35);
            animation: genesisCoreBlink 1.25s ease-in-out infinite;
        }

        @keyframes genesisCoreBlink {
            0%,100% { opacity: 1; transform: scale(1); }
            50% { opacity: .25; transform: scale(.72); }
        }

        .genesis-sensor-node {
            position: absolute;
            width: 54px;
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: #0c0e0d;
            border: 1px solid rgba(196,122,60,.22);
            color: #b9b0a6;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: .5px;
            box-shadow: 0 0 20px rgba(0,0,0,.42);
            animation: genesisNodeFloat 4s ease-in-out infinite;
        }

        .genesis-node-bme { top: 7%; left: 50%; transform: translateX(-50%); }
        .genesis-node-mq135 { right: 5%; top: 48%; animation-delay: .8s; }
        .genesis-node-mq7 { bottom: 8%; left: 50%; transform: translateX(-50%); animation-delay: 1.5s; }
        .genesis-node-env { left: 5%; top: 48%; animation-delay: 2.1s; }

        @keyframes genesisNodeFloat {
            0%,100% { margin-top: 0; }
            50% { margin-top: -7px; }
        }

        .genesis-sensor-caption {
            position: absolute;
            bottom: 4%;
            left: 50%;
            transform: translateX(-50%);
            color: #69655f;
            font-size: 8px;
            letter-spacing: 2px;
            white-space: nowrap;
        }

        .genesis-home-start {
            max-width: 1180px;
            margin: 20px auto 0;
        }

        .genesis-home-start-note {
            margin-top: 11px;
            text-align: center;
            color: #66625d;
            font-size: 9px;
            letter-spacing: 1.2px;
        }

        @media (max-width: 850px) {
            .genesis-home-hero {
                grid-template-columns: 1fr;
                text-align: center;
                padding: 44px 26px;
            }

            .genesis-home-line {
                margin-left: auto;
                margin-right: auto;
            }

            .genesis-home-description {
                margin-left: auto;
                margin-right: auto;
            }

            .genesis-home-data {
                margin-left: auto;
                margin-right: auto;
            }

            .genesis-sensor-field {
                width: min(350px,90vw);
            }
        }

        </style>

        <section class="genesis-home-wrap">

            <div class="genesis-home-hero">

                <div class="genesis-home-copy">

                    <div class="genesis-home-status">
                        <span class="genesis-home-status-dot"></span>
                        RESEARCH SYSTEM • ONLINE
                    </div>

                    <div class="genesis-home-eyebrow">
                        ARTIFICIAL OLFACTORY SYSTEM
                    </div>

                    <div class="genesis-home-title">
                        GENESIS
                    </div>

                    <div class="genesis-home-line"></div>

                    <div class="genesis-home-tagline">
                        Small Signals. Smarter Warnings.
                    </div>

                    <div class="genesis-home-description">

                        A research-oriented Artificial Olfactory System
                        that combines multiple gas-response signals with
                        environmental measurements and uses machine learning
                        to study measurable response patterns.

                        <br><br>

                        <strong>
                            GENESIS turns individual sensor responses into
                            a combined computational fingerprint.
                        </strong>

                        The trained model compares that fingerprint with
                        predefined patterns within controlled research data.

                    </div>

                    <div class="genesis-home-data">

                        <div class="genesis-home-data-cell">
                            <span class="genesis-home-data-value">03</span>
                            <span class="genesis-home-data-label">
                                GAS-RESPONSE<br>SENSING ELEMENTS
                            </span>
                        </div>

                        <div class="genesis-home-data-cell">
                            <span class="genesis-home-data-value">06</span>
                            <span class="genesis-home-data-label">
                                MEASURED<br>INPUT FEATURES
                            </span>
                        </div>

                        <div class="genesis-home-data-cell">
                            <span class="genesis-home-data-value">ML</span>
                            <span class="genesis-home-data-label">
                                PATTERN<br>RECOGNITION
                            </span>
                        </div>

                    </div>

                </div>

                <div class="genesis-sensor-field">

                    <div class="genesis-orbit"></div>
                    <div class="genesis-orbit genesis-orbit-two"></div>

                    <div class="genesis-sensor-node genesis-node-bme">
                        BME688
                    </div>

                    <div class="genesis-sensor-node genesis-node-mq135">
                        MQ-135
                    </div>

                    <div class="genesis-sensor-node genesis-node-mq7">
                        MQ-7
                    </div>

                    <div class="genesis-sensor-node genesis-node-env">
                        ENV
                    </div>

                    <div class="genesis-home-core">
                        G
                    </div>

                    <div class="genesis-sensor-caption">
                        MULTI-SENSOR RESPONSE FIELD
                    </div>

                </div>

            </div>

        </section>
        """)


        # ====================================================
        # START ANALYSIS
        # ====================================================

        with gr.Column(elem_classes="genesis-home-start"):

            start_btn = gr.Button(
                "Start Analysis  →",
                variant="primary",
                elem_classes="analyze-btn"
            )

            gr.HTML("""
            <div class="genesis-home-start-note">
                ENTER THE SIX-DIMENSIONAL RESPONSE PROFILE TO BEGIN
            </div>
            """)


        # ====================================================
        # WHY GENESIS
        # ====================================================

        gr.HTML("""
        <div style="
            text-align:center;
            max-width:850px;
            margin:56px auto 30px;
        ">

            <div class="page-badge">
                WHY GENESIS?
            </div>

            <div class="section-title">
                Why Look for Small Signals?
            </div>

            <div class="section-description">

                Modern sensing systems can capture many small
                changes that may be difficult to interpret individually.
                GENESIS explores whether combining multiple responses
                can reveal meaningful patterns through machine learning.

            </div>

        </div>
        """)


        with gr.Row():

            gr.HTML("""
            <div class="info-card">

                <div class="info-number">01</div>

                <div class="info-heading">
                    Multiple Signals
                </div>

                <div class="info-text">
                    Several sensor responses are analysed together
                    to create a richer response profile.
                </div>

            </div>
            """)


            gr.HTML("""
            <div class="info-card">

                <div class="info-number">02</div>

                <div class="info-heading">
                    Pattern Recognition
                </div>

                <div class="info-text">
                    Machine learning compares new response profiles
                    with patterns learned from the research dataset.
                </div>

            </div>
            """)


            gr.HTML("""
            <div class="info-card">

                <div class="info-number">03</div>

                <div class="info-heading">
                    Early Warning
                </div>

                <div class="info-text">
                    The research concept focuses on recognising
                    patterns that may deserve further investigation.
                </div>

            </div>
            """)


        # ====================================================
        # ASK GENESIS
        # ====================================================

        gr.HTML("""
        <div class="query-box">

            <div class="section-title">
                Explore GENESIS
            </div>

            <div class="section-description">

                Ask about the sensors, machine-learning model,
                pattern recognition or the research concept.

            </div>

        </div>
        """)


        with gr.Row():

            query_input = gr.Textbox(
                placeholder="Ask something about GENESIS...",
                show_label=False,
                scale=5
            )

            query_btn = gr.Button(
                "Ask",
                elem_classes="analyze-btn",
                scale=1
            )


        query_output = gr.HTML("""
        <div class="query-answer">
            Ask a question to learn more about GENESIS.
        </div>
        """)


        gr.HTML("""
        <div class="disclaimer">

            <b>Research Notice:</b>

            GENESIS is a simulated proof-of-concept research
            prototype. Its output represents machine-learning
            pattern classification and must not be interpreted
            as a medical diagnosis.

        </div>
        """)


    # ========================================================
    # ANALYSIS
    # ========================================================

    with gr.Column(

        visible=False,

        elem_classes="page"

    ) as analysis_page:


        gr.HTML("""

        <div style="text-align:center;">

            <div class="page-badge">
                STEP 01
            </div>

            <div class="visual-title">
                Sensor Analysis
            </div>

            <div class="visual-description">

                Enter the simulated six-dimensional
                sensor-response profile to begin
                computational pattern recognition.

            </div>

        </div>

        """)


        with gr.Row():


            with gr.Column(
                elem_classes="panel"
            ):

                gr.HTML("""

                <div class="section-title">
                    Sensor Response Input
                </div>

                <div class="section-description">

                    Enter values representing the submitted
                    research sample.

                </div>

                """)


                bme688 = gr.Number(

                    label="BME688 Gas Response",

                    value=0.70

                )


                mq135 = gr.Number(

                    label="MQ-135 Response",

                    value=0.30

                )


                mq7 = gr.Number(

                    label="MQ-7 Response",

                    value=0.65

                )


                temperature = gr.Number(

                    label="Temperature (°C)",

                    value=24.5

                )


                humidity = gr.Number(

                    label="Humidity (%)",

                    value=55.0

                )


                pressure = gr.Number(

                    label="Pressure (hPa)",

                    value=1008.0

                )


                analyze_button = gr.Button(

                    "Analyze Sample  →",

                    variant="primary",

                    elem_classes="analyze-btn"

                )


            with gr.Column(
                elem_classes="panel"
            ):

                gr.HTML("""

                <div class="section-title">
                    How GENESIS Works
                </div>

                <div class="section-description">

                    The six measurements form a combined
                    response fingerprint. The trained model
                    compares this fingerprint against learned
                    response patterns.

                </div>

                <div class="pipeline-flow">

                    <div class="pipeline-node">
                        Sample
                    </div>

                    <div class="pipeline-arrow">
                        →
                    </div>

                    <div class="pipeline-node">
                        Sensors
                    </div>

                    <div class="pipeline-arrow">
                        →
                    </div>

                    <div class="pipeline-node">
                        Response Pattern
                    </div>

                    <div class="pipeline-arrow">
                        →
                    </div>

                    <div class="pipeline-node">
                        AI / ML
                    </div>

                    <div class="pipeline-arrow">
                        →
                    </div>

                    <div class="pipeline-node">
                        Result
                    </div>

                </div>

                """)


    # ========================================================
    # RESULT
    # ========================================================

    with gr.Column(

        visible=False,

        elem_classes="page"

    ) as result_page:


        gr.HTML("""

        <div style="text-align:center;">

            <div class="page-badge">
                STEP 02
            </div>

            <div class="visual-title">
                Analysis Result
            </div>

            <div class="visual-description">

                Computational classification of the submitted
                sensor-response profile.

            </div>

        </div>

        """)


        prediction_output = gr.HTML("""

        <div class="result-card">

            <div class="result-label">
                GENESIS CLASSIFICATION
            </div>

            <div class="pattern-result">
                Awaiting Analysis
            </div>

            <div class="confidence">

                Model Confidence

                <strong>—</strong>

            </div>

            <div class="status">
                SYSTEM READY
            </div>

        </div>

        """)


        gr.HTML("""

        <div class="section-title"
             style="margin-top:34px;">

            Pattern Distribution

        </div>

        """)


        probability_output = gr.HTML(

            "No analysis performed yet."

        )


    # ========================================================
    # BAR GRAPH
    # ========================================================

    with gr.Column(

        visible=False,

        elem_classes="page"

    ) as bar_page:


        gr.HTML("""

        <div style="text-align:center;">

            <div class="page-badge">
                VISUALIZATION 01
            </div>

            <div class="visual-title">
                Pattern Distribution
            </div>

            <div class="visual-description">

                Comparative model probabilities across
                the recognised response-pattern classes.

            </div>

        </div>

        """)


        bar_plot = gr.HTML("""

        <div class="graph-card">

            <div class="graph-heading">
                Awaiting Analysis
            </div>

            <div class="graph-subheading">
                Run an analysis to generate the live probability graph.
            </div>

        </div>

        """)


    # ========================================================
    # PIE GRAPH
    # ========================================================

    with gr.Column(

        visible=False,

        elem_classes="page"

    ) as pie_page:


        gr.HTML("""

        <div style="text-align:center;">

            <div class="page-badge">
                VISUALIZATION 02
            </div>

            <div class="visual-title">
                Pattern Composition
            </div>

            <div class="visual-description">

                Proportional representation of the model's
                response-pattern distribution.

            </div>

        </div>

        """)


        pie_plot = gr.HTML("""

        <div class="graph-card">

            <div class="graph-heading">
                Awaiting Analysis
            </div>

            <div class="graph-subheading">
                Run an analysis to generate the live composition graph.
            </div>

        </div>

        """)


    # ========================================================
    # SENSOR PROFILE
    # ========================================================

    with gr.Column(

        visible=False,

        elem_classes="page"

    ) as profile_page:


        gr.HTML("""

        <div style="text-align:center;">

            <div class="page-badge">
                VISUALIZATION 03
            </div>

            <div class="visual-title">
                Sensor Response Profile
            </div>

            <div class="visual-description">

                Visual representation of the six submitted
                sensor and environmental measurements.

            </div>

        </div>

        """)


        profile_plot = gr.HTML("""

        <div class="graph-card">

            <div class="graph-heading">
                Awaiting Analysis
            </div>

            <div class="graph-subheading">
                Run an analysis to generate the live sensor profile.
            </div>

        </div>

        """)


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""

    <div class="footer">

        <strong>GENESIS RESEARCH PROTOTYPE</strong>

        <br><br>

        Artificial Olfactory System •
        Machine-Learning Pattern Recognition •
        Simulated Proof-of-Concept Data

        <br><br>

        GENESIS identifies patterns within the research dataset.
        It does not diagnose disease.

    </div>

    """)


    # ========================================================
    # PAGE LIST
    # ========================================================

    pages = [

        home_page,
        analysis_page,
        result_page,
        bar_page,
        pie_page,
        profile_page

    ]


    # ========================================================
    # NAVIGATION
    # ========================================================

    home_btn.click(

        lambda: show_page("home"),

        outputs=pages

    )


    analysis_btn.click(

        lambda: show_page("analysis"),

        outputs=pages

    )


    result_btn.click(

        lambda: show_page("result"),

        outputs=pages

    )


    bar_btn.click(

        lambda: show_page("bar"),

        outputs=pages

    )


    pie_btn.click(

        lambda: show_page("pie"),

        outputs=pages

    )


    profile_btn.click(

        lambda: show_page("profile"),

        outputs=pages

    )


    start_btn.click(

        lambda: show_page("analysis"),

        outputs=pages

    )


    # ========================================================
    # QUERY
    # ========================================================

    query_btn.click(

        fn=answer_query,

        inputs=query_input,

        outputs=query_output

    )


    query_input.submit(

        fn=answer_query,

        inputs=query_input,

        outputs=query_output

    )


    # ========================================================
    # FINAL ANALYSIS FUNCTION
    #
    # THIS IS THE IMPORTANT FIX.
    #
    # The model probabilities are converted into a NORMALIZED
    # dictionary using the exact keys expected by:
    #
    # create_bar_html()
    # create_pie_html()
    # probability display
    #
    # This fixes the problem where the Result showed values
    # but the Bar Graph showed 0%.
    # ========================================================

    def final_run_analysis(

        bme,
        mq135_value,
        mq7_value,
        temp,
        hum,
        press

    ):


        # ----------------------------------------------------
        # Convert inputs safely
        # ----------------------------------------------------

        bme = float(bme)

        mq135_value = float(
            mq135_value
        )

        mq7_value = float(
            mq7_value
        )

        temp = float(temp)

        hum = float(hum)

        press = float(press)


        # ----------------------------------------------------
        # RUN EXISTING TRAINED GENESIS MODEL
        # ----------------------------------------------------
        # Use the SAME feature order and scaler from Cells 3-4.
        # This fixes the callback without changing any page.
        # ----------------------------------------------------

        X_new = pd.DataFrame(
            [[
                bme,
                mq135_value,
                mq7_value,
                temp,
                hum,
                press
            ]],
            columns=feature_columns
        )


        X_scaled = scaler.transform(
            X_new
        )


        prediction_encoded = model.predict(
            X_scaled
        )[0]


        probabilities_array = model.predict_proba(
            X_scaled
        )[0]


        # Convert the Random Forest's encoded classes
        # back to the original Pattern_A / Pattern_B / etc.
        class_labels = label_encoder.inverse_transform(
            model.classes_
        )


        model_probabilities = {

            str(label): float(probability)

            for label, probability in zip(
                class_labels,
                probabilities_array
            )

        }


        predicted_pattern = str(
            label_encoder.inverse_transform(
                [prediction_encoded]
            )[0]
        )


        confidence = float(
            max(probabilities_array)
        )


        # ----------------------------------------------------
        # NORMALIZE MODEL OUTPUT
        # ----------------------------------------------------

        probabilities = {

            "Pattern_A": 0.0,

            "Pattern_B": 0.0,

            "Pattern_C": 0.0,

            "Pattern_D": 0.0,

            "Unknown": 0.0

        }


        for key, value in model_probabilities.items():

            clean_key = str(key).strip().lower()

            numeric_value = float(value)


            if numeric_value > 1:

                numeric_value /= 100


            numeric_value = max(
                0.0,
                min(
                    1.0,
                    numeric_value
                )
            )


            if clean_key in [
                "pattern_a",
                "pattern a"
            ]:

                probabilities["Pattern_A"] = numeric_value


            elif clean_key in [
                "pattern_b",
                "pattern b"
            ]:

                probabilities["Pattern_B"] = numeric_value


            elif clean_key in [
                "pattern_c",
                "pattern c"
            ]:

                probabilities["Pattern_C"] = numeric_value


            elif clean_key in [
                "pattern_d",
                "pattern d"
            ]:

                probabilities["Pattern_D"] = numeric_value


            elif clean_key == "unknown":

                probabilities["Unknown"] = numeric_value


        # ----------------------------------------------------
        # CLEAN PREDICTION
        # ----------------------------------------------------

        clean_prediction = clean_pattern_name(

            predicted_pattern

        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_value = float(
            confidence
        )


        if confidence_value > 1:

            confidence_value /= 100


        confidence_value = max(
            0.0,
            min(
                1.0,
                confidence_value
            )
        )


        confidence_percent = (
            confidence_value * 100
        )


        # ----------------------------------------------------
        # RESULT CARD — VISUAL ONLY
        # ----------------------------------------------------

        # Self-contained styling so NO other page is changed.
        prediction_html = f"""

        <style>

        .genesis-result-visual {{
            max-width: 1080px;
            margin: 5px auto 0;
            padding: 36px;
            border-radius: 24px;
            background:
                radial-gradient(circle at 78% 18%, rgba(88,183,154,.09), transparent 26%),
                radial-gradient(circle at 20% 80%, rgba(196,122,60,.09), transparent 30%),
                linear-gradient(145deg, #151817, #0b0d0c);
            border: 1px solid rgba(196,122,60,.19);
            box-shadow:
                0 28px 80px rgba(0,0,0,.36),
                inset 0 1px rgba(255,255,255,.04);
            position: relative;
            overflow: hidden;
        }}

        .genesis-result-visual::before {{
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -110px;
            top: -130px;
            border-radius: 50%;
            border: 1px solid rgba(196,122,60,.11);
            box-shadow:
                0 0 0 22px rgba(196,122,60,.03),
                0 0 0 46px rgba(196,122,60,.016);
            pointer-events: none;
        }}

        .genesis-result-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin-bottom: 26px;
        }}

        .genesis-result-label {{
            color: #d7cfc5;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 3px;
        }}

        .genesis-result-live {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #8ccbb5;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 1.5px;
        }}

        .genesis-result-live-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #63d6a8;
            box-shadow: 0 0 8px #63d6a8, 0 0 15px rgba(99,214,168,.35);
            animation: genesisResultBlink 1.45s ease-in-out infinite;
        }}

        @keyframes genesisResultBlink {{
            0%,100% {{ opacity: .35; transform: scale(.82); }}
            50% {{ opacity: 1; transform: scale(1.15); }}
        }}

        .genesis-result-content {{
            display: grid;
            grid-template-columns: 1.08fr .92fr;
            align-items: center;
            gap: 38px;
        }}

        .genesis-result-pattern-kicker {{
            color: #77736e;
            font-size: 9px;
            letter-spacing: 2px;
            margin-bottom: 9px;
        }}

        .genesis-result-pattern {{
            font-size: clamp(38px, 5vw, 60px);
            font-weight: 850;
            letter-spacing: 1px;
            line-height: 1;
            color: #f1eee9;
        }}

        .genesis-result-pattern span {{
            color: #e0a064;
        }}

        .genesis-result-note {{
            margin-top: 15px;
            color: #88837c;
            font-size: 10.5px;
            line-height: 1.75;
            max-width: 520px;
        }}

        .genesis-confidence-orbit {{
            width: 205px;
            height: 205px;
            margin: 0 auto;
            border-radius: 50%;
            padding: 13px;
            background: conic-gradient(#e0a064 {confidence_percent:.2f}%, rgba(255,255,255,.07) 0);
            box-shadow: 0 0 45px rgba(196,122,60,.11);
            animation: genesisConfidenceIn .8s cubic-bezier(.2,.8,.2,1) both;
        }}

        @keyframes genesisConfidenceIn {{
            from {{ opacity: 0; transform: scale(.72) rotate(-12deg); }}
            to {{ opacity: 1; transform: scale(1) rotate(0deg); }}
        }}

        .genesis-confidence-inner {{
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: #0f1110;
            border: 1px solid rgba(255,255,255,.07);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        .genesis-confidence-number {{
            font-size: 37px;
            line-height: 1;
            font-weight: 850;
            color: #f1eee9;
        }}

        .genesis-confidence-symbol {{
            color: #e0a064;
            font-size: 17px;
            vertical-align: top;
        }}

        .genesis-confidence-caption {{
            margin-top: 9px;
            color: #77736e;
            font-size: 8px;
            letter-spacing: 1.8px;
        }}

        .genesis-result-footer {{
            margin-top: 30px;
            padding-top: 17px;
            border-top: 1px solid rgba(255,255,255,.07);
            display: flex;
            justify-content: space-between;
            gap: 20px;
            color: #716c65;
            font-size: 9px;
            line-height: 1.5;
        }}

        .genesis-result-footer strong {{
            color: #a69f96;
        }}

        @media (max-width: 760px) {{
            .genesis-result-content {{
                grid-template-columns: 1fr;
                text-align: center;
            }}

            .genesis-result-note {{
                margin-left: auto;
                margin-right: auto;
            }}

            .genesis-result-top,
            .genesis-result-footer {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}

        </style>

        <div class="genesis-result-visual">

            <div class="genesis-result-top">

                <div class="genesis-result-label">
                    GENESIS CLASSIFICATION
                </div>

                <div class="genesis-result-live">
                    <span class="genesis-result-live-dot"></span>
                    ANALYSIS COMPLETE
                </div>

            </div>


            <div class="genesis-result-content">

                <div>

                    <div class="genesis-result-pattern-kicker">
                        PREDICTED RESEARCH PATTERN
                    </div>

                    <div class="genesis-result-pattern">
                        {clean_prediction}
                    </div>

                    <div class="genesis-result-note">
                        The model assigned this input profile its highest
                        learned class probability. This percentage reflects
                        model confidence for the classification, not a medical diagnosis.
                    </div>

                </div>


                <div class="genesis-confidence-orbit">

                    <div class="genesis-confidence-inner">

                        <div class="genesis-confidence-number">
                            {confidence_percent:.2f}<span class="genesis-confidence-symbol">%</span>
                        </div>

                        <div class="genesis-confidence-caption">
                            MODEL CONFIDENCE
                        </div>

                    </div>

                </div>

            </div>


            <div class="genesis-result-footer">

                <div>
                    <strong>STATUS</strong><br>
                    Pattern recognition finished successfully.
                </div>

                <div style="text-align:right;">
                    <strong>INTERPRETATION</strong><br>
                    Research classification only • No diagnosis
                </div>

            </div>

        </div>

        """


        # ----------------------------------------------------
        # PROBABILITY DISPLAY
        # ----------------------------------------------------

        probability_html = ""


        for label, key in [

            ("Pattern A", "Pattern_A"),

            ("Pattern B", "Pattern_B"),

            ("Pattern C", "Pattern_C"),

            ("Pattern D", "Pattern_D"),

            ("Unknown", "Unknown")

        ]:


            value = probabilities.get(
                key,
                0.0
            )


            percentage = value * 100


            probability_html += f"""

            <div class="probability-row">

                <div class="probability-label">

                    <span>
                        {label}
                    </span>

                    <span>
                        {percentage:.2f}%
                    </span>

                </div>

                <div class="probability-track">

                    <div
                        class="probability-fill"
                        style="
                            width:{percentage:.2f}%;
                        "
                    ></div>

                </div>

            </div>

            """


        # ====================================================
        # CREATE PREMIUM VISUALS
        #
        # SAME probability values are passed to the existing
        # Bar Graph / Pie Graph / Sensor Profile functions.
        # ====================================================

        bar_html = create_bar_html(

            {
                key: value * 100
                for key, value
                in probabilities.items()
            }

        )


        pie_html = create_pie_html(

            {
                key: value * 100
                for key, value
                in probabilities.items()
            }

        )


        profile_html = create_profile_html(

            bme,

            mq135_value,

            mq7_value,

            temp,

            hum,

            press

        )


        # ----------------------------------------------------
        # MOVE TO RESULT PAGE
        # ----------------------------------------------------

        page_updates = show_page(
            "result"
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return (

            prediction_html,

            probability_html,

            bar_html,

            pie_html,

            profile_html,

            page_updates[1],

            page_updates[2]

        )

    # ========================================================
    # CONNECT ANALYSIS
    # ========================================================

    analyze_button.click(

        fn=final_run_analysis,

        inputs=[

            bme688,

            mq135,

            mq7,

            temperature,

            humidity,

            pressure

        ],

        outputs=[

            prediction_output,

            probability_output,

            bar_plot,

            pie_plot,

            profile_plot,

            analysis_page,

            result_page

        ]

    )


# ============================================================
# READY
# ============================================================

print(
    "🧬 GENESIS PREMIUM INTERFACE READY."
)

print(
    "✓ Home"
)

print(
    "✓ Analysis"
)

print(
    "✓ Result"
)

print(
    "✓ Premium Animated Bar Graph"
)

print(
    "✓ Premium Animated Pie Graph"
)

print(
    "✓ Premium Animated Sensor Profile"
)

print(
    "✓ Shared Analysis Values"
)

print(
    "✓ Bar Graph Values Fixed"
)

print(
    "✓ Google Colab Ready"
)


# ============================================================
# LAUNCH
# ============================================================

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
