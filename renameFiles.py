import os

# Remove specific prefixes from all files in a folder
def rename_files(folder_path):
    # Durchlaufen Sie jede Datei im Ordner
    for filename in os.listdir(folder_path):
        if filename[:4] == 'PXL_':
            # Konstruieren Sie den neuen Dateinamen ohne die ersten vier Zeichen
            new_filename = filename[4:]

            # Erstellen Sie den vollständigen Pfad für alte und neue Dateinamen
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            # Umbenennen Sie die Datei
            os.rename(old_filepath, new_filepath)

# Geben Sie den Pfad zu Ihrem Ordner an
folder_path = 'Pictures3D/train/3/'

# Rufen Sie die Funktion auf
rename_files(folder_path)