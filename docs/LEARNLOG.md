# LEARNLOG – was wir gebaut haben und warum

Für das Team, das neu in Computer Vision ist. Pro Meilenstein ein paar Sätze und die Fachbegriffe.

## M0 – Generator und Bewertung verstehen

Bevor wir Code schreiben, haben wir `generate_dataset.py` gelesen (Ergebnis: `docs/NOTES.md`). Weil
der Testsatz aus genau diesem Code kommt, müssen wir nichts raten: Schriftgröße 20 px, 28 px Abstand
zwischen Zellen, Zahlen rechtsbündig, Labels linksbündig, feste Wortlisten. Die Baseline erreicht 0.348.
**Begriffe:** *Ground Truth* (die richtige Antwort), *TEDS* (Ähnlichkeit zweier Tabellen als Baum),
*TEDS-S* (nur die Struktur, ohne Text).

## M1 – Durchgehendes Gerüst

`python -m tablesalt.run --images DIR --out DIR` schreibt für jedes Bild sofort eine JSON-Datei. Stürzt
eine Tabelle ab, springt die Baseline ein, notfalls `[]` – ein Fehler beendet nie den Lauf.
`tools/eval_local.py` misst alles mit einem Befehl und trägt das Ergebnis in `docs/RESULTS.md` ein.
**Begriffe:** *Fallback*, *atomares Schreiben* (erst temporäre Datei, dann umbenennen).

## M2 – Preprocessing

Wir teilen das Bild durch eine geschätzte Hintergrundhelligkeit (morphologisches *Closing* = Text
„zuschmieren“) – Schatten und graue Kopfzeilen verschwinden. Dann *Binarisierung* mit *Otsu* (Schwellwert
automatisch). Die Schräglage finden wir mit einem *Projektionsprofil*: Wir drehen das Bild probeweise und
nehmen den Winkel, bei dem die Zeilensummen am „spitzesten“ sind. Für die *Perspektive* nutzen wir, dass
Labels links und Zahlen rechts bündig stehen: Diese Kanten sind im Original gerade Linien. Vier Linien
(links, rechts, erste und letzte Zeile) ergeben vier Ecken → *Homographie* (`cv2.getPerspectiveTransform`).
Lehre: Die Ausgabefläche muss aus den transformierten Bildecken berechnet werden, sonst schneidet man
Teile der Tabelle ab.

## M3/M4 – Struktur (grid, hlines, borderless)

Statt Linien zu suchen, finden wir *Blobs*: Wir verschmieren den Text horizontal (Dilatation um ~0.85 ×
Schrifthöhe). Wörter einer Zelle verschmelzen, Nachbarzellen (≥ 28 px Abstand) nicht → ein Blob pro Zelle.
Zeilen = Gruppen ähnlicher y-Werte. Spalten = gemeinsame rechte Kanten der Zahlen (rechtsbündig!); bei
`grid` direkt die senkrechten Linien. Das funktioniert für alle drei Stile gleich.
Lehre: Verschwommene Wörter bilden lange waagrechte Pixel-Läufe und sahen wie Tabellenlinien aus – echte
Linien müssen fast so breit wie die Tabelle sein.
**Begriffe:** *Dilatation*, *Opening* (dünne Strukturen entfernen), *Connected Components*, *RANSAC*.

## M5 – Kopfzeilen und verbundene Zellen

Zwei Kopfzeilen erkennt man am „Position/Item/Metric“-Label, das genau zwischen zwei Zeilen steht
(`rowspan=2`), oder an Gruppennamen (Actual, Plan, …). Die Gruppen teilen die Spalten 1..n-1 auf; welche
Aufteilung passt, bestimmt eine *dynamische Programmierung* (jedes Label muss mittig über seinen Spalten
stehen). `—` und `see note 3` sind im Generator immer über 2 Spalten verbunden, `–` nie.

## M6 – Zell-OCR

Jeder Blob wird ausgeschnitten, auf ~40 px Höhe skaliert und an **einen** Tesseract-Prozess pro Tabelle
übergeben (Bildliste statt 100 Prozessstarts). Jeder Blob wird zweimal gelesen (enger und weiter
Ausschnitt), weil jeder Ausschnitt andere Fehler macht: eng schneidet die flache Oberkante der „5“ ab
(→ „9“), weit nimmt Störpunkte mit. `--psm 7` (eine Textzeile) liefert manchmal gar nichts; dann liest
`--psm 13` (rohe Zeile) nach.
**Begriffe:** *PSM* (Page Segmentation Mode), *Konfidenz*.

## M7 – Postprocessing

Pro Tabelle gibt es genau ein Zahlenformat. Per *Mehrheitsentscheid* bestimmen wir es und drucken jede
Zahl aus ihren Ziffern neu (en/de immer 2 Nachkommastellen). Labels werden mit *Fuzzy-Matching*
(`rapidfuzz`) auf die Wortlisten des Generators gezogen; Spaltenköpfe sind immer die ersten n-1 Einträge
einer Liste. Ein fehlendes Minus erkennt die Geometrie (kurzer flacher Strich links im Blob).

## M8 – Docker

`submission_template/Dockerfile` installiert Tesseract + gepinnte Python-Pakete und kopiert nur
`tablesalt/`, `baselines/`, `submission_template/`. Ein Worker pro CPU, `OMP_THREAD_LIMIT=1`.
