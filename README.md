# plans-MOB — Plan PDF + photos → maison 3D animée

## Fichiers
- `plan.pdf` — plans (1 page = 1 étage)
- `photos/` — photos des façades (nord/sud) et de l'intérieur (salon, cuisine)
- `plan_to_3d.py` — plan PDF → `maison.glb` (murs + ouvertures découpées + escalier)
- `render_anim.py` — `maison` 3D texturée + `animation.gif` (visite 360°)

## Utilisation (Mac)
```bash
python3 -m pip install opencv-python-headless trimesh numpy pymupdf matplotlib pillow manifold3d
python3 plan_to_3d.py plan.pdf      # -> maison.glb + debug_dessus.png
python3 render_anim.py plan.pdf     # -> animation.gif (36 images, 360°)
open animation.gif
```

## Réglages
- Échelle : `PIXELS_PAR_METRE = 132` dans `plan_to_3d.py` (cote 12,12 m = 1595 px)
- Textures : dictionnaire `TEXTURES` dans `render_anim.py` (nord/sud); côtés = bardage gris
- Pour ajouter une façade est/ouest : mettre la photo dans `photos/` et ajouter une entrée dans `TEXTURES`

## Limites
- Détection de murs par vision (OpenCV) : quelques cloisons fines/portes peuvent manquer.
- Rendu : moteur maison (peintre + homographie) — pour un rendu pro, on pourra brancher Blender.
