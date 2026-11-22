import cv2
import numpy as np

def masque_murs(image_path, seuil=200, close_px=15, epaisseur_px=9):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaire = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((close_px, close_px), np.uint8)
    ferme = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, k)
    dist = cv2.distanceTransform(ferme, cv2.DIST_L2, 5)
    masque = np.uint8(dist > epaisseur_px) * 255
    # garder uniquement le plus gros amas connexe (= le batiment), elimine cotes/cartouche
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(masque, 8)
    if nb > 1:
        plus_gros = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        masque = np.uint8(labels == plus_gros) * 255
    return img, masque

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
        voisins = 0
        for t in segments:
            if t is s:
                continue
            a2 = np.degrees(np.arctan2(-(t[3] - t[1]), t[2] - t[0])) % 180.0
            da = min(abs(angle - a2), 180 - abs(angle - a2))
            r2 = (t[0] + t[2]) / 2 * nx + (t[1] + t[3]) / 2 * ny
            if da < 6 and abs(rho - r2) < tol_dist:
                voisins += 1
        if voisins <= max_voisins:
            out.append(s)
    return out

def detecter(img, masque, min_ratio=0.035):
    diag = np.hypot(*img.shape[:2])
    lines = cv2.HoughLinesP(masque, 1, np.pi / 180, threshold=40,
                            minLineLength=int(diag * min_ratio), maxLineGap=20)
    if lines is None:
        return []
    segs = [tuple(map(int, l)) for l in lines.reshape(-1, 4)]
    murs = []
    for (x1, y1, x2, y2) in segs:
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if min(angle, 90 - angle) > 10:
            continue
        murs.append((x1, y1, x2, y2))
    return murs

for page in ["plan_page0.png", "plan_page1.png"]:
    img, masque = masque_murs(page)
    segs = nettoyer_groupes(fusionner(detecter(img, masque)))
    vis = img.copy()
    for s in segs:
        cv2.line(vis, (s[0], s[1]), (s[2], s[3]), (0, 0, 255), 4)
    out = page.replace("plan_page", "debugv5")
    cv2.imwrite(out, vis)
    L = max((np.hypot(s[2]-s[0], s[3]-s[1]) for s in segs), default=0)
    xs = [v for s in segs for v in (s[0], s[2])]; ys = [v for s in segs for v in (s[1], s[3])]
    print(f"{page}: {len(segs)} murs, plus long={int(L)}px, bbox={(max(xs)-min(xs))}x{(max(ys)-min(ys))}px -> {out}")
