from ultralytics import YOLO
from pathlib import Path

# 1. Chemins
IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Charger le modèle pré-entraîné (~52 Mo, téléchargé automatiquement)
model = YOLO("yolov8m-seg.pt")

# 3. Lister les images disponibles
images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.jpeg"))
print(f"Images trouvées : {len(images)}")

# 4. Lancer la segmentation sur les 5 premières images
for img_path in images[:5]:
    # --- Début du bloc indenté ---
    results = model.predict(
        source=str(img_path),
        conf=0.25,        # ignorer les détections < 25% de confiance
        save=True,        # sauvegarder les images avec les masques
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True
    )
    
    for r in results:
        # Chaque ligne ici doit aussi être décalée par rapport au "for r in results"
        labels = [model.names[int(c)] for c in r.boxes.cls]
        print(f"{img_path.name} -> {set(labels)}")
    # --- Fin du bloc indenté ---

print("Terminé. Résultats dans data/results/predict/")