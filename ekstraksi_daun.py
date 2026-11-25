import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from skimage.feature import graycomatrix, graycoprops

# ==================================================
# PARAMETER
# ==================================================
IMG_SIZE = 128
DATASET_PATH = "dataset"

# ==================================================
# EKSTRAKSI FITUR WARNA
# ==================================================
def extract_color(image):

    (B, G, R) = cv2.split(image)

    return [

        np.mean(R),
        np.mean(G),
        np.mean(B),

        np.std(R),
        np.std(G),
        np.std(B)

    ]

# ==================================================
# EKSTRAKSI FITUR TEKSTUR
# ==================================================
def extract_texture(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    glcm = graycomatrix(
        gray,
        [1],
        [0],
        256,
        symmetric=True,
        normed=True
    )

    return [

        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'dissimilarity')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0],
        graycoprops(glcm, 'energy')[0][0],
        graycoprops(glcm, 'correlation')[0][0],
        graycoprops(glcm, 'ASM')[0][0]

    ]

# ==================================================
# EKSTRAKSI FITUR BENTUK
# ==================================================
def extract_shape(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / h if h != 0 else 0

        extent = area / (w * h) if (w * h) != 0 else 0

        hull = cv2.convexHull(cnt)

        solidity = (
            area / cv2.contourArea(hull)
            if cv2.contourArea(hull) != 0
            else 0
        )

    else:

        area = 0
        perimeter = 0
        aspect_ratio = 0
        extent = 0
        solidity = 0

    edge_count = np.sum(edges)

    return [

        area,
        perimeter,
        aspect_ratio,
        extent,
        solidity,
        edge_count

    ]

# ==================================================
# VISUALISASI BEFORE/AFTER EKSTRAKSI FITUR
# ==================================================
def show_one(img_display, judul, nama_fitur, filename, label, cmap=None):
    """Tampilkan 1 figure: original (kiri) vs hasil fitur (kanan)."""

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    fig.suptitle(
        f'Kelas: {label.upper()} | File: {filename}',
        fontsize=11,
        fontweight='bold'
    )

    img_rgb_orig = cv2.cvtColor(img_display["original"], cv2.COLOR_BGR2RGB)

    axes[0].imshow(img_rgb_orig)
    axes[0].set_title('Original')
    axes[0].axis('off')

    if cmap:
        axes[1].imshow(img_display["result"], cmap=cmap, aspect='auto')
    else:
        axes[1].imshow(img_display["result"])

    axes[1].set_title(judul)
    axes[1].axis('off')

    plt.tight_layout()

    nama_file_vis = (
        f"vis_{label}_"
        f"{nama_fitur}_"
        f"{os.path.splitext(filename)[0]}.png"
    )

    plt.savefig(nama_file_vis, bbox_inches='tight')

    plt.show()

    print(f"  Tersimpan: {nama_file_vis}")


def visualize_extraction(image, filename, label):

    basename = os.path.splitext(filename)[0]

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (B, G, R) = cv2.split(image)

    blank = np.zeros_like(B)

    # ---------- FITUR WARNA (6 fitur) ----------

    # 1. R_mean
    img_R = cv2.cvtColor(
        cv2.merge([blank, blank, R]),
        cv2.COLOR_BGR2RGB
    )
    show_one(
        {"original": image, "result": img_R},
        f'R_mean = {np.mean(R):.2f}',
        'R_mean', filename, label
    )

    # 2. G_mean
    img_G = cv2.cvtColor(
        cv2.merge([blank, G, blank]),
        cv2.COLOR_BGR2RGB
    )
    show_one(
        {"original": image, "result": img_G},
        f'G_mean = {np.mean(G):.2f}',
        'G_mean', filename, label
    )

    # 3. B_mean
    img_B = cv2.cvtColor(
        cv2.merge([B, blank, blank]),
        cv2.COLOR_BGR2RGB
    )
    show_one(
        {"original": image, "result": img_B},
        f'B_mean = {np.mean(B):.2f}',
        'B_mean', filename, label
    )

    # 4. R_std
    show_one(
        {"original": image, "result": R},
        f'R_std = {np.std(R):.2f}',
        'R_std', filename, label,
        cmap='Reds'
    )

    # 5. G_std
    show_one(
        {"original": image, "result": G},
        f'G_std = {np.std(G):.2f}',
        'G_std', filename, label,
        cmap='Greens'
    )

    # 6. B_std
    show_one(
        {"original": image, "result": B},
        f'B_std = {np.std(B):.2f}',
        'B_std', filename, label,
        cmap='Blues'
    )

    # ---------- FITUR TEKSTUR (6 fitur) ----------
    glcm = graycomatrix(
        gray, [1], [0], 256,
        symmetric=True, normed=True
    )

    glcm_img = glcm[:, :, 0, 0]
    glcm_img = (
        glcm_img / glcm_img.max() * 255
    ).astype(np.uint8)

    props = {
        'contrast':      'contrast',
        'dissimilarity': 'dissimilarity',
        'homogeneity':   'homogeneity',
        'energy':        'energy',
        'correlation':   'correlation',
        'ASM':           'ASM'
    }

    for nama, prop in props.items():
        nilai = graycoprops(glcm, prop)[0][0]
        show_one(
            {"original": image, "result": glcm_img},
            f'{nama} = {nilai:.4f}',
            nama, filename, label,
            cmap='hot'
        )

    # ---------- FITUR BENTUK (6 fitur) ----------
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_contour = img_rgb.copy()

    if len(contours) > 0:

        cnt = max(contours, key=cv2.contourArea)

        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        extent = area / (w * h) if (w * h) != 0 else 0
        hull = cv2.convexHull(cnt)
        solidity = (
            area / cv2.contourArea(hull)
            if cv2.contourArea(hull) != 0
            else 0
        )
        edge_count = np.sum(edges)

        cv2.drawContours(
            img_contour, [cnt], -1,
            (255, 0, 0), 2
        )

    else:
        area = perimeter = aspect_ratio = 0
        extent = solidity = edge_count = 0

    bentuk_fitur = {
        'area':         (edges, f'area = {area:.2f}',               'gray'),
        'perimeter':    (img_contour, f'perimeter = {perimeter:.2f}', None),
        'aspect_ratio': (img_contour, f'aspect_ratio = {aspect_ratio:.4f}', None),
        'extent':       (img_contour, f'extent = {extent:.4f}',      None),
        'solidity':     (img_contour, f'solidity = {solidity:.4f}',  None),
        'edge_count':   (edges, f'edge_count = {int(edge_count)}',   'gray'),
    }

    for nama, (result_img, judul, cmap) in bentuk_fitur.items():
        show_one(
            {"original": image, "result": result_img},
            judul,
            nama, filename, label,
            cmap=cmap
        )

# ==================================================
# LOAD DATASET
# ==================================================
print("Memuat dataset daun tomat...")

labels_list = [
    "sehat",
    "busuk",
    "bakteri"
]

data = []

# Simpan 1 sampel per kelas untuk divisualisasi
visualized_labels = set()

print("\nVisualisasi ekstraksi fitur (1 sampel per kelas):")

for label in labels_list:

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    for file in os.listdir(folder):

        path = os.path.join(
            folder,
            file
        )

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        # Visualisasi 1 sampel pertama per kelas
        if label not in visualized_labels:
            visualize_extraction(img, file, label)
            visualized_labels.add(label)

        fitur = (

            extract_color(img) +

            extract_texture(img) +

            extract_shape(img)

        )

        data.append(
            [file] + fitur + [label]
        )

# ==================================================
# DATAFRAME
# ==================================================
cols = [

    "nama_file",

    "R_mean",
    "G_mean",
    "B_mean",

    "R_std",
    "G_std",
    "B_std",

    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",

    "area",
    "perimeter",
    "aspect_ratio",
    "extent",
    "solidity",
    "edge_count",

    "label"
]

df = pd.DataFrame(
    data,
    columns=cols
)

# ==================================================
# SIMPAN EXCEL
# ==================================================
df.to_excel(
    "hasil_ekstraksi_daun_tomat.xlsx",
    index=False
)

print("\nFile Excel berhasil disimpan!")

# ==================================================
# PREPARE DATA
# ==================================================
X = df.drop(
    ["nama_file", "label"],
    axis=1
)

y = df["label"]

# ==================================================
# STANDARDISASI
# ==================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# SPLIT DATA
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42

)

# ==================================================
# VISUALISASI JUMLAH DATA
# ==================================================
label_counts = (
    y.value_counts()
    .reset_index()
)

label_counts.columns = [
    'label',
    'count'
]

plt.figure(figsize=(8, 5))

ax = sns.barplot(

    data=label_counts,

    x='label',

    y='count',

    palette='muted'

)

total = label_counts['count'].sum()

for bar in ax.patches:

    tinggi = bar.get_height()

    persen = (tinggi / total) * 100

    ax.text(

        bar.get_x() + bar.get_width()/2,

        tinggi + 1,

        f'{int(tinggi)}\n({persen:.1f}%)',

        ha='center',

        fontsize=10,

        fontweight='bold'

    )

plt.title('Jumlah Data per Kelas Daun Tomat')

plt.xlabel("Kelas Daun Tomat")

plt.ylabel("Jumlah Data")

plt.savefig(
    'visual_jumlah_data.png',
    bbox_inches='tight'
)

plt.show()

# ==================================================
# PCA VISUALISASI
# ==================================================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(

    x=X_pca[:, 0],

    y=X_pca[:, 1],

    hue=y,

    palette={
        "sehat": "green",
        "busuk": "red",
        "bakteri": "blue"
    },

    s=40,

    alpha=0.7

)

plt.title('Sebaran Data PCA - Dataset Daun Tomat')

plt.xlabel("PC 1")

plt.ylabel("PC 2")

plt.savefig(
    'visual_sebaran_pca.png',
    bbox_inches='tight'
)

plt.show()

# ==================================================
# MODEL MACHINE LEARNING
# ==================================================
models = {

    "SVM": SVC(
        kernel='linear',
        probability=True,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB()

}

results = {}

# ==================================================
# TRAINING & EVALUASI
# ==================================================
print("\n====================================")
print("AKURASI MODEL")
print("====================================")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(f"\n===== {name} =====")

    print(
        f"Akurasi: "
        f"{acc:.4f} "
        f"({(acc*100):.2f}%)"
    )

    print("\nClassification Report:")

    print(

        classification_report(
            y_test,
            y_pred
        )

    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    plt.figure(figsize=(8, 6))

    warna_cmap = (

        'Blues'
        if name == 'SVM'

        else 'Greens'
        if name == 'Random Forest'

        else 'Oranges'
        if name == 'KNN'

        else 'Purples'

    )

    sns.heatmap(

        cm,

        annot=True,

        fmt='d',

        cmap=warna_cmap,

        xticklabels=model.classes_,

        yticklabels=model.classes_

    )

    plt.title(
        f'Confusion Matrix - {name}\n'
        f'Akurasi: {(acc*100):.2f}%'
    )

    plt.ylabel('Aktual')

    plt.xlabel('Prediksi')

    nama_file_cm = (
        f"confusion_matrix_"
        f"{name.replace(' ', '_')}.png"
    )

    plt.savefig(
        nama_file_cm,
        bbox_inches='tight'
    )

    plt.show()

# ==================================================
# PERBANDINGAN AKURASI SEMUA MODEL
# ==================================================
plt.figure(figsize=(10, 6))

model_names = list(results.keys())

acc_values = list(results.values())

x = np.arange(len(model_names))

bars = plt.bar(

    x,

    acc_values,

    color=['steelblue', 'seagreen', 'darkorange', 'mediumpurple'],

    width=0.5

)

for bar in bars:

    yval = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        yval + 0.01,

        f"{yval*100:.1f}%",

        ha='center'

    )

plt.xticks(x, model_names)

plt.ylabel("Akurasi")

plt.title("Perbandingan Akurasi Model Klasifikasi Daun Tomat")

plt.ylim(0, 1.1)

plt.savefig(
    'perbandingan_akurasi.png',
    bbox_inches='tight'
)

plt.show()

# ==================================================
# MODEL TERBAIK
# ==================================================
best_model = max(
    results,
    key=results.get
)

print("\n====================================")
print("SEMUA PROSES SELESAI!")
print("====================================")

print(
    f"\nModel Terbaik: {best_model}"
)

print(
    f"Akurasi Tertinggi: "
    f"{results[best_model]*100:.2f}%"
)

print(
    "\nVisualisasi berhasil disimpan:"
)

print("- hasil_ekstraksi_daun_tomat.xlsx")
print("- vis_<kelas>_<fitur>_<nama_file>.png (18 file per kelas x 3 kelas = 54 file)")
print("- visual_jumlah_data.png")
print("- visual_sebaran_pca.png")
print("- confusion_matrix_SVM.png")
print("- confusion_matrix_Random_Forest.png")
print("- confusion_matrix_KNN.png")
print("- confusion_matrix_Naive_Bayes.png")
print("- perbandingan_akurasi.png")