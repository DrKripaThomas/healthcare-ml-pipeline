# Healthcare Dataset: Pima Indians Diabetes Database

## 1. Provenance and Source
- **Original Source**: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), National Institutes of Health (NIH), Bethesda, Maryland, USA.
- **Archive Hosting**: UCI Machine Learning Repository / Kaggle Public Teaching Datasets.
- **Study Context**: All patients represented in this public teaching dataset are female individuals aged 21 years and older of Pima Indian heritage living near Phoenix, Arizona.
- **Academic Purpose**: This dataset is utilized strictly for educational, analytical, and health informatics benchmarking. **It is not real-time clinical electronic health record (EHR) data, and it must never be used to make unverified clinical diagnostic decisions.**

---

## 2. Feature Dictionary & Clinical Measurement Units

| Column Name | Data Type | Units / Description | Normal Clinical Reference Range |
| :--- | :--- | :--- | :--- |
| **`Pregnancies`** | Integer | Number of times pregnant | $\ge 0$ (0 is biologically valid) |
| **`Glucose`** | Numeric | Plasma glucose concentration at 2 hours in an oral glucose tolerance test (OGTT) (mg/dL) | Normal: 70–99 mg/dL (fasting), < 140 mg/dL (2-hr OGTT) |
| **`BloodPressure`** | Numeric | Diastolic blood pressure (mm Hg) | Normal: 60–80 mm Hg |
| **`SkinThickness`** | Numeric | Triceps skin fold thickness (mm), surrogate for subcutaneous adiposity | Normal female: 18–25 mm |
| **`Insulin`** | Numeric | 2-Hour serum insulin ($\mu\text{U/mL}$) | Normal fasting: 16–166 $\mu\text{U/mL}$ |
| **`BMI`** | Numeric | Body Mass Index: $\text{weight (kg)} / (\text{height (m)})^2$ | Normal: 18.5–24.9; Overweight: 25.0–29.9; Obese: $\ge 30.0$ |
| **`DiabetesPedigreeFunction`** | Numeric | Genetic risk score synthesizing diabetes history in relatives | Score $> 0.0$; higher values indicate stronger hereditary risk |
| **`Age`** | Integer | Age in calendar years | Range in cohort: 21–81 years |
| **`Outcome`** | Integer (Binary) | Target variable: Clinical diabetes diagnostic status | `0` = Negative (Non-diabetic), `1` = Positive (Diabetic) |

---

## 3. Critical Health Informatics Consideration: Biological Zero Measurements

In general non-medical tabular datasets, a value of `0` is often treated as a standard numerical value. However, in **clinical physiological data**, several zero entries are physiologically implausible:
- **`Glucose = 0`**: Incompatible with conscious human life (severe fatal hypoglycemia).
- **`BloodPressure = 0`**: Represents lack of systemic arterial pressure (cardiac arrest).
- **`SkinThickness = 0`**: Absence of subcutaneous fat and skin (anatomically impossible).
- **`Insulin = 0`**: Biologically abnormal in non-diabetic living patients during oral tolerance testing.
- **`BMI = 0`**: Zero body mass or infinite height (physically impossible).

### Audit of Zero Values in This Dataset:
- `Pregnancies`: 111 zeros (**Valid** biological measurement: nulliparous women).
- `Glucose`: 5 zeros (**Missing / unrecorded**).
- `BloodPressure`: 35 zeros (**Missing / unrecorded**).
- `SkinThickness`: 227 zeros (**Missing / unrecorded**).
- `Insulin`: 374 zeros (**Missing / unrecorded**).
- `BMI`: 11 zeros (**Missing / unrecorded**).
- `DiabetesPedigreeFunction`: 0 zeros.
- `Age`: 0 zeros.
- `Outcome`: 500 negatives (65.1%), 268 positives (34.9%).

### Treatment & Data Leakage Prevention:
To adhere strictly to health data preprocessing standards and avoid data leakage:
1. Measurements recorded as `0` in `['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']` are treated as missing values (`NaN`).
2. The dataset is **split into training and testing partitions first** (`stratify=y`, `random_state=42`).
3. An imputer (`SimpleImputer(strategy='median')`) is fit **solely on the training set** to replace missing entries with training medians (chosen for robustness against physiological skewness), and then applied to test and inference data.
