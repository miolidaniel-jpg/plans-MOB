import sys
import cv2
import numpy as np
import trimesh
import pymupdf

PIXELS_PAR_METRE = 132.0   # 1595 px = 12,12 m (plus longue facade, cote lue sur le plan)
EPAISSEUR_MUR = 0.20
HAUTEUR_MUR = 2.50


def masque_murs(img, seuil=200, close_px=15, epaisseur_px=9):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaire = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY_INV)
    ferme = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    dist = cv2.distanceTransform(ferme, cv2.DIST_L2, 5)
    return np.uint8(dist > epaisseur_px) * 255, ferme


def composant_batiment(ferme):
    dil = cv2.dilate(ferme, np.ones((25, 25), np.uint8))
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i


def fusionner(segments, tol_angle=6, tol_dist=30):
    reps = []
    for (x1, y1, x2, y2) in segments:
        longueur = float(np.hypot(x2 - x1, y2 - y1))
        angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180.0
        nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
        rho = (x1 + x2) / 2 * nx + (y1 + y2) / 2 * ny
        for r in reps:
            da = min(abs(angle - r["angle"]), 180 - abs(angle - r["angle"]))
            if da < tol_angle and abs(rho - r["rho"]) < tol_dist:
                if longueur > r["longueur"]:
                    r.update(angle=angle, rho=rho, x1=x1, y1=y1, x2=x2, y2=y2, longueur=longueur)
                break
        else:
            reps.append(dict(angle=angle, rho=rho, x1=x1, y1=y1, x2=x2, y2=y2, longueur=longueur))
    return [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in reps]


def nettoyer_groupes(segments, tol_dist=45, max_voisins=3):
    out = []
    for s in segments:
        x1, y1, x2, y2 = s
        angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180.0
        nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
        rho = (x1 + x2) / 2 * nx + (y1 + y2) / 2 * ny
        v = 0
        for t in segments:
            if t is s:
                continue
            a2 = np.degrees(np.arctan2(-(t[3] - t[1]), t[2] - t[0])) % 180.0
            if min(abs(angle - a2), 180 - abs(angle - a2)) < 6 and abs(rho - ((t[0]+t[2])/2*nx + (t[1]+t[3])/2*ny)) < tol_dist:
                v += 1
        if v <= max_voisins:
            out.append(s)
    return out


def detecter_murs(img):
    masque, ferme = masque_murs(img)
    comp = composant_batiment(ferme)
    diag = np.hypot(*img.shape[:2])
    lines = cv2.HoughLinesP(masque, 1, np.pi / 180, threshold=40,
                            minLineLength=int(diag * 0.035), maxLineGap=20)
    segs = [] if lines is None else [tuple(map(int, l)) for l in lines.reshape(-1, 4)]
    murs = []
    for (x1, y1, x2, y2) in segs:
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if min(angle, 90 - angle) > 10:
            continue
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if not comp[my, mx]:
            continue
        murs.append((x1, y1, x2, y2))
    return nettoyer_groupes(fusionner(murs))


def segment_vers_boite(x1, y1, x2, y2, z0, ox, oy):
    s = 1.0 / PIXELS_PAR_METRE
    longueur = np.hypot(x2 - x1, y2 - y1) * s
    angle = np.arctan2(-(y2 - y1), x2 - x1)
    cx, cy = ((x1 + x2) / 2 - ox) * s, ((y1 + y2) / 2 - oy) * s
    box = trimesh.creation.box(extents=[longueur + EPAISSEUR_MUR, EPAISSEUR_MUR, HAUTEUR_MUR])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    box.apply_translation([cx, cy, z0 + HAUTEUR_MUR / 2])
    return box


def pdf_vers_maison(pdf_path, sortie="maison.glb", debug_png="debug_dessus.png"):
    doc = pymupdf.open(pdf_path)
    meshes = []
    dessus = None
    for etage, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        segs = detecter_murs(img)
        print(f"étage {etage} (page {etage}) : {len(segs)} murs détectés")
        z0 = etage * HAUTEUR_MUR
        # origine commune = coin du composant bâtiment (alignement des étages)
        masque, ferme = masque_murs(img)
        dil = cv2.dilate(ferme, np.ones((25, 25), np.uint8))
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ox, oy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        floor = [segment_vers_boite(*s, z0=z0, ox=ox, oy=oy) for s in segs]
        if not floor:
            continue
        xs = [v[0] for m in floor for v in m.vertices]
        ys = [v[1] for m in floor for v in m.vertices]
        dalle = trimesh.creation.box(extents=[max(xs) - min(xs) + 0.4, max(ys) - min(ys) + 0.4, 0.2])
        dalle.apply_translation([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, z0 - 0.1])
        meshes += floor + [dalle]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if dessus is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            dessus = (fig, ax)
        fig, ax = dessus
        for s in segs:
            ax.plot([(s[0]-ox)/PIXELS_PAR_METRE, (s[2]-ox)/PIXELS_PAR_METRE],
                    [-(s[1]-oy)/PIXELS_PAR_METRE, -(s[3]-oy)/PIXELS_PAR_METRE],
                    lw=3, label=f"étage {etage}" if etage == 0 else None)
    if dessus:
        fig, ax = dessus
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(True); ax.legend()
        fig.savefig(debug_png, dpi=110)
        plt.close(fig)
        print(f"vue de dessus -> {debug_png}")
    maison = trimesh.util.concatenate(meshes)
    maison.export(sortie)
    b = maison.bounds
    print(f"Modèle exporté : {sortie}")
    print(f"Dimensions : {b[1,0]-b[0,0]:.2f} m x {b[1,1]-b[0,1]:.2f} m x {b[1,2]-b[0,2]:.2f} m (L x l x H)")


if __name__ == "__main__":
    pdf_vers_maison(sys.argv[1] if len(sys.argv) > 1 else "plan.pdf")
