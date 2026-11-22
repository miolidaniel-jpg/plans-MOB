import sys, math
import cv2
import numpy as np
import trimesh
import pymupdf
import plan_to_3d as P

PIXELS = P.PIXELS_PAR_METRE
TEXTURES = {
    "nord": "photos/facade nord.jpeg",
    "sud": "photos/facade sud.JPG",
}
COULEURS = {"mur": (225, 220, 210), "cote": (112, 112, 118), "toit": (60, 60, 65), "sol": (150, 110, 70), "interieur": (235, 235, 230)}


def detecter_page(page):
    pix = page.get_pixmap(dpi=200)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    masque, binaire, ferme = P.masque_murs(img)
    comp, stats = P.composant_batiment(ferme)
    ox, oy = stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP]
    segs = P.detecter_murs(img, masque, comp)
    ouvs = P.trouver_ouvertures(segs)
    escs = P.detecter_escaliers(binaire, comp)
    return segs, ouvs, escs, ox, oy


def construire_faces(pdf_path):
    doc = pymupdf.open(pdf_path)
    faces = []  # (pts3d [3,3], texture_key or color)
    for etage, page in enumerate(doc):
        segs, ouvs, escs, ox, oy = detecter_page(page)
        z0 = etage * P.HAUTEUR_MUR
        murs = [P.segment_vers_boite(*s, z0=z0, ox=ox, oy=oy) for s in segs]
        for ouv in ouvs:
            cx, cy = ouv["centre"]
            best, bd = None, 1e9
            for s, m in zip(segs, murs):
                x1, y1, x2, y2 = s
                n = max(np.hypot(x2 - x1, y2 - y1), 1)
                ux, uy = (x2 - x1) / n, (y2 - y1) / n
                t = max(0, min(n, (cx - x1) * ux + (cy - y1) * uy))
                d = np.hypot(cx - (x1 + ux * t), cy - (y1 + uy * t))
                if d < bd:
                    bd, best = d, m
            if best is not None and bd < 150:
                murs[murs.index(best)] = P.decouper_ouverture(best, ouv, z0, ox, oy)
        for e in escs:
            murs += P.construire_marches(e[:4], e[4], z0, ox, oy)
        xs = [v[0] for m in murs for v in m.vertices]
        ys = [v[1] for m in murs for v in m.vertices]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        dalle = trimesh.creation.box(extents=[xmax - xmin + 0.4, ymax - ymin + 0.4, 0.2])
        dalle.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, z0 - 0.1])
        meshes = murs + [dalle]
        if etage == len(doc) - 1:
            toit = trimesh.creation.box(extents=[xmax - xmin + 1.2, ymax - ymin + 1.2, 0.3])
            toit.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, z0 + P.HAUTEUR_MUR + 0.15])
            meshes.append(toit)
        for m in meshes:
            v, f = m.vertices, m.faces
            for tri in f:
                pts = v[tri]
                n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                n = n / (np.linalg.norm(n) + 1e-9)
                c = pts.mean(axis=0)
                key = None
                if abs(n[2]) < 0.1:  # face verticale = mur
                    if abs(n[1]) > 0.9:
                        key = "nord" if c[1] > (ymax + ymin) / 2 else "sud"
                    elif abs(n[0]) > 0.9:
                        key = "cote"
                else:
                    key = "toit" if n[2] > 0 and c[2] > z0 + P.HAUTEUR_MUR - 0.5 else "sol"
                color = COULEURS.get(key, COULEURS["interieur"])
                tex = TEXTURES.get(key) if key in TEXTURES else None
                faces.append((pts, tex, color))
    return faces


def projeter(pts, cam, W, H, f):
    rel = pts - cam["pos"]
    fwd = cam["fwd"] / np.linalg.norm(cam["fwd"])
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    x = rel @ right; y = rel @ up; z = rel @ fwd
    z = np.maximum(z, 0.1)
    return np.stack([W / 2 + f * x / z, H / 2 - f * y / z], axis=1), z.mean()


def rendre(faces, W=800, H=450, f=650, n_frames=36, sortie="animation.gif"):
    from PIL import Image
    allpts = np.vstack([p for p, _, _ in faces])
    centre = allpts.mean(axis=0)
    R = np.linalg.norm(allpts.max(axis=0) - allpts.min(axis=0)) * 0.9
    frames = []
    for k in range(n_frames):
        a = 2 * math.pi * k / n_frames
        cam = {"pos": centre + np.array([R * math.cos(a), R * math.sin(a), R * 0.45]),
               "fwd": centre - (centre + np.array([R * math.cos(a), R * math.sin(a), R * 0.45]))}
        frame = np.full((H, W, 3), 200, np.uint8)  # ciel gris clair
        # sol
        cv2.rectangle(frame, (0, int(H * 0.75)), (W, H), (120, 140, 110), -1)
        items = []
        for pts, tex, color in faces:
            p2d, z = projeter(pts, cam, W, H, f)
            items.append((z, p2d, tex, color))
        items.sort(key=lambda t: -t[0])
        for z, p2d, tex, color in items:
            poly = p2d.astype(np.int32)
            if tex:
                timg = cv2.imread(tex)
                th, tw = timg.shape[:2]
                src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
                p2d4 = np.vstack([p2d, p2d[-1:]]) if len(p2d) == 3 else p2d
                Hm = cv2.getPerspectiveTransform(src, p2d4.astype(np.float32))
                x0 = max(int(p2d[:, 0].min()), 0); y0 = max(int(p2d[:, 1].min()), 0)
                x1 = min(int(p2d[:, 0].max()) + 1, W); y1 = min(int(p2d[:, 1].max()) + 1, H)
                bw, bh = x1 - x0, y1 - y0
                if bw <= 0 or bh <= 0:
                    continue
                T = np.float32([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
                warped = cv2.warpPerspective(timg, T @ Hm, (bw, bh))
                mask = np.zeros((bh, bw), np.uint8)
                cv2.fillPoly(mask, [poly - np.array([x0, y0])], 255)
                sub = frame[y0:y1, x0:x1]
                sub[mask > 0] = warped[mask > 0]
            else:
                bgr = (color[2], color[1], color[0])
                cv2.fillPoly(frame, [poly], bgr)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    frames[0].save(sortie, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"animation -> {sortie} ({n_frames} images)")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "plan.pdf"
    faces = construire_faces(pdf)
    print(f"{len(faces)} faces a rendre")
    rendre(faces, sortie="animation.gif")
