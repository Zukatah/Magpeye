# Magpeye

## Magpeye - Allgemeine Infos

Magpeye ist ein Projekt mit zwei Zielen / aus zwei Teilen:
(1) Es soll ein ML-Model trainiert werden, welches aus Videomaterial von Badmintoncourts erkennen kann, wann Federbälle mit dem Boden kollidieren und auf welcher Seite der Spiellinien dies geschieht, um Entscheidungen darüber zu treffen, welcher Spieler bzw. welches Team einen Ballwechsel gewonnen hat.
(2) Es soll eine (Android-)App entwickelt werden, welche das in (1) erwähnte Model verwendet, um in Echtzeit die in (1) beschriebenen Entscheidungen treffen zu können.

Angelehnt an das im Tennis verbreitete "Hawkeye"-System heißt dieses Projekt "Magpeye".

Dieser Ordner bzw. dieses Repository enthält Teil (1) des Projektes. Die (Android-)App wird in einem separaten Ordner/Repository gepflegt.


## Magpeye - Aufbau des Projektes

Die Aufnahme der Videos, das Preprocessing der Videos und das Trainieren des Models funktionieren folgendermaßen:

1. Bei der Aufnahme muss die Smartphonekamera so positioniert sein, dass genau eine Linie komplett sichtbar ist. Diese muss in der unteren Mitte des Bildes beginnen und in der oberen Mitte des Bildes enden. Die Aufnahme muss mit FullHD-Auflösung erfolgen (1920x1080) und bei konstanten (!) 60 FPS. Viele Smartphones können keine konstante Aufnahmerate garantieren (sie unterstützen nur "variable frame rate"). Aufnahmen von solchen Smartphones eignen sich nicht für die Aufnahme von Trainingsmaterial für dieses Model, da Predictions für Kollisionen nicht möglich sind, wenn wiederholt ein oder mehrere aufeinanderfolgende Frames fehlen und somit das Zeitintervall zwischen zwei aufeinanderfolgenden Frames unbekannt ist und stark variieren kann. Die Videodateien legen wir alle in den Ordner Videos_Raw.

2. Mithilfe des ffmpeg-Programms bearbeiten wir die aufgenommenen Videos. Dadurch reduzieren wir den Speicherplatz um ca. 90%, entfernen beschädigte Frames und fügen Timestamps in die einzelnen Frames der Videos ein, um danach Frames, in denen Kollisionen des Federballs mit dem Boden stattfinden, labeln zu können. Die nötigen Befehle lauten:
./ffmpeg -i absoluterPfadZurOriginalvideodatei.mp4 -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099: fontsize=72" -y absoluterSpeicherpfadDerZuErstellendenVideodatei.mp4
./ffmpeg -i absoluterPfadZurMitVorigemBefehlErstelltenVideodatei.mp4 -vf mpdecimate,setpts=N/FRAME_RATE/TB absoluterSpeicherpfadDerZuErstellendenVideodatei.mp4
Die nach dem zweiten Schritt erstellte Videodatei muss mit "_Frames_RD.mp4" enden. Die Endung signalisiert dem nächsten Script, dass das Preprocessing vollständig umgesetzt wurde. Alle auf "_Frames_RD.mp4" endenden (also alle fertig preprocessten) Videodateien verschieben wir von Videos_Raw in Videos_Preprocessed.

3. Als nächstes müssen die Videodateien gelabelt werden. Hierzu legen wir für jede Videodatei in Videos_Preprocessed eine CSV-Datei im selben Ordner an.
Die zur Videodatei MeinVideo_Frames_RD.mp4 gehörende CSV-Datei heißt MeinVideo_Notes.csv.
Wir notieren in jeder CSV-Datei nun die Kollisionen in der zugehörigen Videodatei. Das geht so:
Findet in einem Frame bspw. der Videodatei MeinVideo_Frames_RD.mp4 eine Kollision des Federballs mit dem Boden statt, so tragen wir die Nummer des entsprechenden Frames in MeinVideo_Notes.csv ein. Wir tragen diese Nummer in Zeile 1 ein, wenn die Kollision links der Feldlinie stattgefunden hat, in Zeile 2, wenn die Kollision auf der Feldlinie stattgefunden hat und in Zeile 3, wenn die Kollision rechts der Feldlinie stattgefunden hat.
Die CSV-Datei MUSS stets 3 Zeilen haben, auch wenn eine oder mehrere der Zeilen keinen Inhalt haben / leer sind.
Haben wir die Nummern aller Frames mit Kollisionen aus MeinVideo_Frames_RD.mp4 in MeinVideo_Notes.csv eingetragen, so speichern wir letztere.

4. Wir führen nun das Script createLabelsFromNotes.py aus. Dieses erzeugt auf Basis der von uns gelabelten CSV-Dateien nun Labeldateien, die für jedes einzelne Frame eines Videos ein Label beinhalten.
Z.B. wird so für MeinVideo_Frames_RD.mp4 auf Basis von MeinVideo_Notes.csv die Label-Datei MeinVideo_Labels.csv erstellt.

5. Jetzt führen wir das Script create3DPicturesFromPreprocessedVideos.py aus.
Dieses erzeugt nun für jedes Video aus Videos_Preprocessed auf Basis der Labeldateien eine Vielzahl von Training Examples im Ordner Pictures3D_NewlyCreated\train.
Ein Training Example ist eine .npy-Datei (npy steht für numpy, ein Python-Mathe-Modul) und besteht aus 5 aufeinanderfolgenden Frames, gespeichert als dreidimensionales Graustufenbild (480x270x5).
Alle Videodateien, für die Training Examples erstellt wurden, verschieben wir anschließend in Videos_AddedToDS.

6. Wir verschieben alle Training Examples von Pictures3D_NewlyCreated in Pictures3D_Input und führen dann das Script split3DImagesIntoTrainAndVal.py aus, welches die Training Examples in ein Training und ein Validation Set teilt, und zwar im Ordner Pictures3D

7. Da durch das Script create3DPicturesFromPreprocessedVideos.py unverhältnismäßig mehr Class_0 (keine Kollision) Training Examples entstehen als von den anderen Klassen, verwenden wir das Script movePercentageOfFiles.py, um einen bestimmten Prozentsatz (je nach Trainingmaterial ca. 90%) der Class_0 Training Examples zu ignorieren. Das Script verschiebt diese einfach in den Ordner Pictures3D_TempStorage_Class0, wo die Training Examples im Trainingsvorgang nicht berücksichtigt werden.

8. Mit dem Script createNewModel.py können wir ein neues Model erstellen - oder mit dem Script loadAndTrainModel.py ein bestehendes laden und trainieren (auf den erstellten Training Examples). Bei bisherigen Trainings haben sich sinkende Learning Rates von 0.001 hin zu 0.000005 bewährt, um Accuracies bei der Class Prediction von ca. 95% zu erreichen.

9. Falsch eingeordnete Training Examples können wir mit dem Script showMisclassified3DImages.py ansehen, um zu sehen, mit welchen Training Examples das Model noch Probleme hat.