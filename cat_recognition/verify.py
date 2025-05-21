from PIL import Image
import os

folder = './train/PetImages/'

for subfolder in os.listdir(folder):
    path = os.path.join(folder, subfolder)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                img = Image.open(file_path)
                img.verify()  # verifică dacă fișierul este o imagine validă
            except Exception:
                print(f"Fișier invalid sau corupt: {file_path}")


# https://claude.ai/share/8634a874-bbbf-4bdd-8187-459c79ab3142