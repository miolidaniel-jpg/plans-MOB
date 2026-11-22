import sys
import cv2
import numpy as np
import trimesh
import pymupdf

PIXELS_PAR_METRE = 132.0
EPAISSEUR_MUR = 0.20
HAUTEUR_MUR = 2.50
SEUIL_PORTE = 40


def masque_murs(img, seuil=200, close_px=15, epaisseur_px=9):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaire = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY_INV)
    ferme = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    dist = cv2.distanceTransform(ferme, cv2.DIST_L2, 5)
    return np.uint8(dist > epaisseur_px) * 255, binaire, ferme


def composant_batiment(ferme):
    dil = cv2.dilate(ferme, np.ones((25, 25), np.uint8))
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i, stats[i]


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


def detecter_murs(img, masque, comp):
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


def lignes_longues(masque):
    diag = np.hypot(*masque.shape[:2])
    lines = cv2.HoughLinesP(masque, 1, np.pi / 180, threshold=40,
                            minLineLength=int(diag * 0.10), maxLineGap=int(diag * 0.35))
    return [] if lines is None else [tuple(map(int, l)) for l in lines.reshape(-1, 4)]


def classer_ouverture(fins, cx, cy, ux, uy, largeur_px):
    nx, ny = -uy, ux
    demi_along, demi_norm = largeur_px / 2 + 25, 150
    mask = np.zeros(fins.shape, np.uint8)
    for dy in range(-demi_norm, demi_norm + 1):
        for dx in range(-int(demi_along), int(demi_along) + 1):
            px, py = int(cx + ux * dx + nx * dy), int(cy + uy * dx + ny * dy)
            if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                mask[py, px] = 255
    ys, xs = np.nonzero(cv2.bitwise_and(fins, mask))
    if len(xs) == 0:
        return "fenetre"
    dist = np.abs((xs - cx) * nx + (ys - cy) * ny)
    return "porte" if dist.max() > SEUIL_PORTE else "fenetre"


def trouver_ouvertures(segments):
    # groupe les murs colineaires (meme ligne) et cherche les trous entre leurs extremites
    groupes = {}
    for (x1, y1, x2, y2) in segments:
        angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180.0
        nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
        rho = (x1 + x2) / 2 * nx + (y1 + y2) / 2 * ny
        for g in groupes.values():
            da = min(abs(angle - g["angle"]), 180 - abs(angle - g["angle"]))
            if da < 6 and abs(rho - g["rho"]) < 30:
                g["segs"].append((x1, y1, x2, y2))
                break
        else:
            groupes[(round(angle), round(rho))] = {"angle": angle, "rho": rho, "segs": [(x1, y1, x2, y2)]}
    ouvertures = []
    for g in groupes.values():
        angle = g["angle"]
        ux, uy = np.cos(np.radians(angle)), -np.sin(np.radians(angle))
        pts = []
        for (x1, y1, x2, y2) in g["segs"]:
            pts.append((x1 * ux + y1 * uy, (x1, y1)))
            pts.append((x2 * ux + y2 * uy, (x2, y2)))
        pts.sort()
        for k in range(len(pts) - 1):
            t1, p1 = pts[k]
            t2, p2 = pts[k + 1]
            gap = t2 - t1
            if 80 <= gap <= 300:
                cx = (p1[0] + p2[0]) / 2
                cy = (p1[1] + p2[1]) / 2
                ouvertures.append({
                    "type": "porte" if gap < 135 else "fenetre",
                    "centre": (cx, cy), "largeur_px": gap,
                    "angle": np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0]),
                })
    return ouvertures


def detecter_escaliers(binaire, comp):
    diag = np.hypot(*binaire.shape[:2])
    lines = cv2.HoughLinesP(binaire, 1, np.pi / 180, threshold=30,
                            minLineLength=int(diag * 0.04), maxLineGap=4)
    if lines is None:
        return []
    segs = [tuple(map(int, l)) for l in lines.reshape(-1, 4)]
    horiz = [s for s in segs if abs(s[3] - s[1]) < 8 and np.hypot(s[2]-s[0], s[3]-s[1]) < diag * 0.3]
    horiz.sort(key=lambda s: s[1])
    groupes, cur = [], []
    for s in horiz:
        if cur and abs(s[1] - cur[-1][1]) < 30:
            cur.append(s)
        else:
            if len(cur) >= 5:
                groupes.append(cur)
            cur = [s]
    if len(cur) >= 5:
        groupes.append(cur)
    out = []
    for g in groupes:
        xs = [v for s in g for v in (s[0], s[2])]
        ys = [v for s in g for v in (s[1], s[3])]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w < diag * 0.5 and h < diag * 0.5:
            mx, my = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2
            if comp[my, mx]:
                out.append((min(xs), min(ys), max(xs), max(ys), len(g)))
    return out


def segment_vers_boite(x1, y1, x2, y2, z0, ox, oy):
    s = 1.0 / PIXELS_PAR_METRE
    longueur = np.hypot(x2 - x1, y2 - y1) * s
    angle = np.arctan2(-(y2 - y1), x2 - x1)
    cx, cy = ((x1 + x2) / 2 - ox) * s, ((y1 + y2) / 2 - oy) * s
    box = trimesh.creation.box(extents=[longueur + EPAISSEUR_MUR, EPAISSEUR_MUR, HAUTEUR_MUR])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    box.apply_translation([cx, cy, z0 + HAUTEUR_MUR / 2])
    return box


def decouper_ouverture(mur, ouv, z0, ox, oy):
    s = 1.0 / PIXELS_PAR_METRE
    cx = (ouv["centre"][0] - ox) * s
    cy = (ouv["centre"][1] - oy) * s
    largeur = ouv["largeur_px"] * s + 0.1
    haut = HAUTEUR_MUR if ouv["type"] == "porte" else 1.2
    z_base = z0 if ouv["type"] == "porte" else z0 + 0.9
    cutter = trimesh.creation.box(extents=[largeur, EPAISSEUR_MUR * 4, haut])
    cutter.apply_transform(trimesh.transformations.rotation_matrix(ouv["angle"], [0, 0, 1]))
    cutter.apply_translation([cx, cy, z_base + haut / 2])
    try:
        return trimesh.boolean.difference([mur, cutter], engine="manifold")
    except Exception:
        return mur


def construire_marches(bbox, n, z0, ox, oy):
    s = 1.0 / PIXELS_PAR_METRE
    x0, y0 = (bbox[0] - ox) * s, (bbox[1] - oy) * s
    x1, y1 = (bbox[2] - ox) * s, (bbox[3] - oy) * s
    w, l = x1 - x0, y1 - y0
    n = max(n, 4)
    marches = []
    if w >= l:
        profondeur = w / n
        for i in range(n):
            h = HAUTEUR_MUR * (i + 1) / n
            m = trimesh.creation.box(extents=[profondeur, l, h])
            m.apply_translation([x0 + profondeur * (i + 0.5), (y0 + y1) / 2, z0 + h / 2])
            marches.append(m)
    else:
        profondeur = l / n
        for i in range(n):
            h = HAUTEUR_MUR * (i + 1) / n
            m = trimesh.creation.box(extents=[w, profondeur, h])
            m.apply_translation([(x0 + x1) / 2, y0 + profondeur * (i + 0.5), z0 + h / 2])
            marches.append(m)
    return marches


def pdf_vers_maison(pdf_path, sortie="maison.glb", debug_png="debug_dessus.png"):
    doc = pymupdf.open(pdf_path)
    meshes = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    for etage, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        masque, binaire, ferme = masque_murs(img)
        comp, stats = composant_batiment(ferme)
        ox, oy = stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP]
        segs = detecter_murs(img, masque, comp)
        fins = cv2.bitwise_and(binaire, cv2.bitwise_not(cv2.dilate(masque, np.ones((7, 7), np.uint8))))
        ouvertures = trouver_ouvertures(segs)
        escaliers = detecter_escaliers(binaire, comp)
        print(f"étage {etage} : {len(segs)} murs, {len(ouvertures)} ouvertures, {len(escaliers)} escalier(s)")
        for o in ouvertures:
            print(f"    - {o['type']} {o['largeur_px']/PIXELS_PAR_METRE:.2f} m")
        z0 = etage * HAUTEUR_MUR
        floor = [segment_vers_boite(*s, z0=z0, ox=ox, oy=oy) for s in segs]
        for ouv in ouvertures:
            cx, cy = ouv["centre"]
            best, bd = None, 1e9
            for s, m in zip(segs, floor):
                x1, y1, x2, y2 = s
                n = max(np.hypot(x2 - x1, y2 - y1), 1)
                ux, uy = (x2 - x1) / n, (y2 - y1) / n
                t = max(0, min(n, (cx - x1) * ux + (cy - y1) * uy))
                d = np.hypot(cx - (x1 + ux * t), cy - (y1 + uy * t))
                if d < bd:
                    bd, best = d, m
            if best is not None and bd < 150:
                floor[floor.index(best)] = decouper_ouverture(best, ouv, z0, ox, oy)
        for e in escaliers:
            floor += construire_marches(e[:4], e[4], z0, ox, oy)
        if not floor:
            continue
        xs = [v[0] for m in floor for v in m.vertices]
        ys = [v[1] for m in floor for v in m.vertices]
        dalle = trimesh.creation.box(extents=[max(xs) - min(xs) + 0.4, max(ys) - min(ys) + 0.4, 0.2])
        dalle.apply_translation([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, z0 - 0.1])
        meshes += floor + [dalle]
        for s in segs:
            ax.plot([(s[0]-ox)/PIXELS_PAR_METRE, (s[2]-ox)/PIXELS_PAR_METRE],
                    [-(s[1]-oy)/PIXELS_PAR_METRE, -(s[3]-oy)/PIXELS_PAR_METRE], lw=3)
        for o in ouvertures:
            ax.plot((o["centre"][0]-ox)/PIXELS_PAR_METRE, -(o["centre"][1]-oy)/PIXELS_PAR_METRE,
                    "o", color="red" if o["type"] == "porte" else "blue", ms=10)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(True)
    fig.savefig(debug_png, dpi=110)
    plt.close(fig)
    print(f"vue de dessus (rouge=porte, bleu=fenetre) -> {debug_png}")
    maison = trimesh.util.concatenate(meshes)
    maison.export(sortie)
    b = maison.bounds
    print(f"Modèle exporté : {sortie}")
    print(f"Dimensions : {b[1,0]-b[0,0]:.2f} m x {b[1,1]-b[0,1]:.2f} m x {b[1,2]-b[0,2]:.2f} m (L x l x H)")


if __name__ == "__main__":
    pdf_vers_maison(sys.argv[1] if len(sys.argv) > 1 else "plan.pdf")
