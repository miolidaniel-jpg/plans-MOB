# plans-MOB — Plan PDF → maison 3D

## Principe
`plan.pdf` (1 page = 1 étage) → détection des murs (OpenCV : seuillage + morphologie + épaisseur + plus gros amas = bâtiment) → extrusion 3D (trimesh) → `maison.glb`.

## Utilisation (sur le Mac)
```bash
python3 -m pip install opencv-python-headless trimesh numpy pymupdf matplotlib
python3 plan_to_3d.py plan.pdf
open maison.glb          # aperçu 3D
```
Sorties : `maison.glb` (modèle 3D 2 étages), `debug_dessus.png` (vue de dessus des murs détectés).

## Échelle
`PIXELS_PAR_METRE = 132` dans `plan_to_3d.py` (calibré sur la cote 12,12 m = 1595 px).

## Limites connues
- Quelques cloisons intérieures fines peuvent manquer ; meubles/cotes/hachures sont filtrés.
- Portes/fenêtres/escalier : non découpés dans les murs pour l'instant (prochaine étape).
