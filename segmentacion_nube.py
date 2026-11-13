import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "FARO_Scan_Full_Sub1_down3_with_normals.ply"


# Clases y colores

CLASSES = {
    0: ("otros", (150, 150, 150)),
    1: ("piso", (30, 60, 230)),
    2: ("techo", (140, 80, 40)),
    3: ("plataforma/losa", (0, 210, 230)),
    4: ("columna", (0, 160, 0)),
    5: ("muro", (220, 220, 220)),
    6: ("viga de concreto", (170, 0, 30)),
    7: ("tuberia oscura", (190, 140, 0)),
    8: ("poste vertical", (150, 255, 90)),
    9: ("equipo/maquinaria", (220, 0, 220)),
    10: ("persona", (255, 140, 190)),
    11: ("vehiculo", (255, 120, 0)),
    12: ("viga metalica", (255, 90, 90)),
    13: ("estructura metalica", (130, 60, 200)),
    14: ("tuberia clara", (255, 225, 60)),
    15: ("tanque", (0, 125, 125)),
}

(OTROS, PISO, TECHO, PLATAFORMA, COLUMNA, MURO, VIGA, TUBERIA, POSTE, EQUIPO,
 PERSONA, VEHICULO, VIGA_METAL, ESTRUCTURA, TUBERIA_CLARA, TANQUE) = range(16)

PALETTE = np.array([color for _, color in CLASSES.values()], dtype=np.uint8)


# Utilidades geometricas

def footprint(Q):
    q = Q[:, :2] - Q[:, :2].mean(0)
    _, v = np.linalg.eigh(q.T @ q)
    extent = np.ptp(q @ v, axis=0)
    return extent[0], extent[1]


def area_xy(Q, cell):
    cells = np.unique(np.floor(Q[:, :2] / cell).astype(np.int64), axis=0)
    return len(cells) * cell ** 2


def fit_circle(p):
    A = np.c_[2 * p, np.ones(len(p))]
    s = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    center = s[:2]
    radius = np.sqrt(max(s[2] + center @ center, 1e-12))
    rms = np.sqrt(((np.linalg.norm(p - center, axis=1) - radius) ** 2).mean())
    return center, radius, rms


def normal_alignment(normals, radial):
    dot = np.abs((normals * radial).sum(1))
    norms = np.linalg.norm(normals, axis=1) * np.linalg.norm(radial, axis=1) + 1e-9
    return np.median(dot / norms)


def vertical_cylinder(P, N, s):
    q = P[s, :2] - P[s, :2].mean(0)
    center, radius, _ = fit_circle(q)
    n, radial = N[s, :2], q - center
    vertical_faces = np.linalg.norm(n, axis=1) > 0.5
    return radius, normal_alignment(n[vertical_faces], radial[vertical_faces])


def chroma(rgb):
    return rgb.max(1) - rgb.min(1)


def brightness(rgb):
    return rgb.max(1)


# Etapa 0: carga y voxelizado

def load_cloud(path, voxel):
    pcd = o3d.io.read_point_cloud(str(path))
    small = pcd.voxel_down_sample(voxel)
    small.normalize_normals()
    return pcd, small


# Etapa 1: descriptores locales y eleccion de eps

def pca_features(P, tree, k, chunk=50_000):
    lam, e1, neighbors = np.zeros((len(P), 3)), np.zeros((len(P), 3)), []
    for start in range(0, len(P), chunk):
        _, nb = tree.query(P[start:start + chunk], k=k, workers=-1)
        Q = P[nb] - P[nb].mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q) / k)
        lam[start:start + chunk] = w[:, ::-1]
        e1[start:start + chunk] = v[:, :, 2]
        neighbors.append(nb)
    return lam, e1, np.vstack(neighbors)


def local_features(P, N, k_small=25, k_large=80):
    tree = cKDTree(P)

    _, nb = tree.query(P, k=k_small, workers=-1)
    same_side = np.sign(np.einsum("nkj,nj->nk", N[nb], N) + 1e-9)[..., None]
    nvar = 1 - np.linalg.norm((N[nb] * same_side).mean(1), axis=1)

    lam, e1, _ = pca_features(P, tree, k_large)
    l1 = np.maximum(lam[:, 0], 1e-12)

    return dict(
        lin=(lam[:, 0] - lam[:, 1]) / l1,
        pla=(lam[:, 1] - lam[:, 2]) / l1,
        sph=lam[:, 2] / l1,
        e1=e1,
        e1z=np.abs(e1[:, 2]),
        nz=np.abs(N[:, 2]),
        nvar=nvar,
        tree=tree,
        nb=nb,
    )


def eps_from_kdistance(tree, P, k, q=95, step=10):
    d, _ = tree.query(P[::step], k=k + 1, workers=-1)
    return float(np.percentile(d[:, -1], q))


def floor_height_map(P, is_floor, cell=1.0):
    F = P[is_floor]
    cells, inv = np.unique(np.floor(F[:, :2] / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    groups = np.split(F[order, 2], np.cumsum(np.bincount(inv))[:-1])
    z_median = np.array([np.median(g) for g in groups])
    _, nearest = cKDTree((cells + 0.5) * cell).query(P[:, :2], workers=-1)
    return z_median[nearest]


# Etapa 2: superficies horizontales (DBSCAN #1)

def stage_horizontal(P, f, labels, voxel):
    idx = np.flatnonzero(f["nz"] > 0.9)
    lab = DBSCAN(eps=2.1 * voxel, min_samples=6).fit_predict(P[idx] * [1, 1, 2.5])

    surfaces = []
    for u in np.unique(lab[lab >= 0]):
        q = P[idx[lab == u]]
        surfaces.append((u, q[:, 2].mean(), area_xy(q, voxel), footprint(q)[0]))

    floor = max(surfaces, key=lambda t: t[2])
    z_top = max(z for _, z, area, _ in surfaces if area > 5)

    for u, z, area, width in surfaces:
        members = idx[lab == u]
        if u == floor[0]:
            labels[members] = PISO
        elif area > 2.0 and z > z_top - 0.8:
            labels[members] = TECHO
        elif area > 3.0 and width > 1.5 and z > floor[1] + 0.3:
            labels[members] = PLATAFORMA

    return floor[1], z_top


# Etapa 3: columnas (DBSCAN #2 en planta)

def stage_columns(P, f, labels, min_height=3.0, G=0.1):
    vertical = (f["nz"] < 0.3) & (labels == OTROS)

    cell = np.floor((P[:, :2] - P[:, :2].min(0)) / G).astype(np.int64)
    key = cell[:, 0] * 1_000_000 + cell[:, 1]
    zbin = np.floor((P[:, 2] - P[:, 2].min()) / 0.25).astype(np.int64)

    occupied = np.unique(np.c_[key[vertical], zbin[vertical]], axis=0)
    keys, n_bins = np.unique(occupied[:, 0], return_counts=True)
    pos = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
    cover = np.where(keys[pos] == key, n_bins[pos] * 0.25, 0.0)

    tall = np.flatnonzero(vertical & (cover >= min_height))
    lab = DBSCAN(eps=2 * G, min_samples=10).fit_predict(P[tall, :2])

    free = np.flatnonzero(labels == OTROS)
    neighborhood = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    found = []

    for u in np.unique(lab[lab >= 0]):
        q = P[tall[lab == u]]
        width, length = footprint(q)

        if np.ptp(q[:, 2]) < min_height:
            continue
        if width > 0.4 and length < 2.0:
            cls = COLUMNA
        elif length >= 2.0 and width < 0.6:
            cls = MURO
        else:
            continue

        cells = np.unique(cell[tall[lab == u]], axis=0)
        cells = (cells[:, None, :] + neighborhood).reshape(-1, 2)
        inside = np.isin(key[free], cells[:, 0] * 1_000_000 + cells[:, 1])
        in_height = (P[free, 2] >= q[:, 2].min() - 0.3) & (P[free, 2] <= q[:, 2].max() + 0.1)
        labels[free[inside & in_height]] = cls
        found.append(cls)

    return found


# Etapa 4: personas y tanques (DBSCAN #3)

def stage_people(P, N, labels, hf, eps, min_pts=8, max_ring=1.0):
    idx = np.flatnonzero((labels == OTROS) & (hf > 0.05) & (hf < 2.4))
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(P[idx])
    groups = [idx[lab == u] for u in np.unique(lab[lab >= 0])]

    others = np.flatnonzero(labels != PISO)
    tree = cKDTree(P[others])
    people = []

    while groups:
        s = groups.pop()
        if len(s) < 50:
            continue

        width, length = footprint(P[s])
        height, base = np.ptp(P[s, 2]), np.percentile(hf[s], 2)
        if not (1.3 <= height <= 2.1 and base < 0.4):
            continue

        if length > 1.0 or width > 0.8:
            if length <= 2.5 and width <= 1.5:
                sub = DBSCAN(eps=0.85 * eps, min_samples=min_pts).fit_predict(P[s])
                if sub.max() >= 1:
                    groups += [s[sub == k] for k in range(sub.max() + 1)]
            continue

        near = others[np.unique(np.concatenate(tree.query_ball_point(P[s], 0.5)))]
        ring = len(np.setdiff1d(near, s)) / len(s)
        radius, align = vertical_cylinder(P, N, s)

        cls = OTROS
        if ring <= max_ring:
            is_tank = 0.15 <= radius <= 0.6 and align >= 0.95
            cls = TANQUE if is_tank else PERSONA
            labels[s] = cls

        people.append(dict(
            idx=s, cls=cls, center=P[s].mean(0), height=height, width=width, length=length,
            ring=ring, R=radius, align=align, accepted=cls == PERSONA,
        ))

    return people


# Etapa 5: tipo de punto (K-means #4)

def stage_point_types(f, labels, k=6, seed=0):
    free = np.flatnonzero(labels == OTROS)
    F = np.c_[f["lin"], f["pla"], f["sph"], f["e1z"], f["nz"], 3 * f["nvar"]][free]
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(F)

    names = []
    for lin, pla, sph, e1z, nz, nvar3 in km.cluster_centers_:
        if sph > 0.3 or nvar3 > 0.5:
            names.append("disperso")
        elif lin > 0.45:
            names.append("lineal " + ("vertical" if e1z > 0.6 else "horizontal"))
        else:
            names.append("planar " + ("horizontal" if nz > 0.6 else "vertical"))

    ptype = np.full(len(labels), "", dtype=object)
    ptype[free] = np.array(names, dtype=object)[km.labels_]
    return ptype, km, names


# Etapa 6: vigas, tuberias y postes (DBSCAN #5)

def direction_features(e):
    r2 = np.sqrt(2)
    return np.c_[
        e[:, 0] ** 2, e[:, 1] ** 2, e[:, 2] ** 2,
        r2 * e[:, 0] * e[:, 1], r2 * e[:, 0] * e[:, 2], r2 * e[:, 1] * e[:, 2],
    ]


def stage_linear(P, N, f, labels, ptype, eps, w_dir=0.6):
    idx = np.flatnonzero((labels == OTROS) & (ptype != "disperso") & (ptype != ""))
    X = np.c_[P[idx], w_dir * direction_features(f["e1"][idx])]
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(X)

    segs = []
    for u in np.unique(lab[lab >= 0]):
        s = idx[lab == u]
        if len(s) < 40:
            continue

        q = P[s] - P[s].mean(0)
        _, v = np.linalg.eigh(q.T @ q)
        proj = q @ v
        length, width = np.ptp(proj[:, 2]), np.ptp(proj[:, 1])
        if length < 1.5 or length < 3 * max(width, 0.05):
            continue

        center, radius, rms = fit_circle(proj[:, :2])
        radial = normal_alignment(N[s] @ v[:, :2], proj[:, :2] - center)
        rel = rms / radius

        circular = radius < 0.6 and (rel < 0.12 or (rel < 0.18 and radial > 0.9))
        vertical = abs(v[2, 2]) > 0.85

        if circular:
            cls = TUBERIA
        elif vertical:
            cls = POSTE
        else:
            cls = VIGA

        labels[s] = cls
        segs.append(dict(
            idx=s, cls=cls, n=len(s), length=length, R=radius, rel=rel, radial=radial,
            z=P[s, 2].mean(), axis=v[:, 2],
        ))

    return segs


# Etapa 7: color de vigas y tuberias (K-means #6 y #7)

def split_by_color(C, labels, segs, cls, new_cls, measure, seed=0):
    group = [s for s in segs if s["cls"] == cls]
    rgb = np.array([np.median(C[s["idx"]], axis=0) for s in group])
    x = measure(rgb).reshape(-1, 1)

    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x)
    high = int(np.argmax(km.cluster_centers_[:, 0]))

    for s, k, value, color in zip(group, km.labels_, x[:, 0], rgb):
        s["color_value"], s["rgb"] = value, color
        if k == high:
            s["cls"] = new_cls
            labels[s["idx"]] = new_cls

    return km


def stage_color(P, C, labels, segs, z_top):
    km_beams = split_by_color(C, labels, segs, VIGA, VIGA_METAL, chroma)
    km_pipes = split_by_color(C, labels, segs, TUBERIA, TUBERIA_CLARA, brightness)

    for s in segs:
        if s["cls"] == TUBERIA and P[s["idx"], 2].max() > z_top - 1.2:
            s["cls"], s["color_value"] = VIGA, chroma(s["rgb"][None])[0]
            labels[s["idx"]] = VIGA

    return km_beams, km_pipes


# Etapa 8: objetos restantes (DBSCAN #8)

def stage_objects(P, labels, hf, nvar, z_top, eps):
    idx = np.flatnonzero(labels == OTROS)
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(P[idx])

    objs = []
    for u in np.unique(lab[lab >= 0]):
        s = idx[lab == u]
        if len(s) < 30:
            continue

        width, length = footprint(P[s])
        height = np.ptp(P[s, 2])
        base = np.percentile(hf[s], 2)
        smooth = np.median(nvar[s])

        is_vehicle = (base < 0.5 and 1.0 <= height <= 2.6 and 3.0 <= length <= 6.5
                      and 1.3 <= width <= 2.6 and smooth < 0.03)

        cls = OTROS
        if is_vehicle:
            cls = VEHICULO
        elif base < 1.0 and len(s) >= 150 and height >= 0.4:
            cls = EQUIPO
        elif base >= 1.0 and len(s) >= 100:
            cls = TECHO if P[s, 2].max() > z_top - 0.6 else ESTRUCTURA

        labels[s] = cls
        objs.append(dict(
            idx=s, cls=cls, n=len(s), width=width, length=length, height=height,
            base=base, smooth=smooth, center=P[s].mean(0),
        ))

    return objs


# Etapa 9: suavizado por votacion de vecinos

def smooth_labels(labels, nb, n_iter=2, min_share=0.6):
    for _ in range(n_iter):
        votes = labels[nb]
        counts = np.stack([(votes == c).sum(1) for c in range(len(CLASSES))], axis=1)
        best = counts.argmax(1)
        wins = counts[np.arange(len(votes)), best] >= min_share * nb.shape[1]
        labels = np.where(wins, best, labels)
    return labels


# Exportacion: PLY y capturas

def save_results(pcd, labels, out_ply):
    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    seg = o3d.geometry.PointCloud(pcd)
    seg.colors = o3d.utility.Vector3dVector(PALETTE[labels] / 255.0)
    o3d.io.write_point_cloud(str(out_ply), seg)

    class_dir = out_ply.parent / "clases"
    class_dir.mkdir(exist_ok=True)
    for c, (name, _) in CLASSES.items():
        idx = np.flatnonzero(labels == c)
        if len(idx):
            fname = f"{c:02d}_{name.replace('/', '_').replace(' ', '_')}.ply"
            o3d.io.write_point_cloud(str(class_dir / fname), seg.select_by_index(idx))

    return seg


def save_snapshots(P, labels, rgb, out_png, n=400_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sample = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False)
    views = [((0, 1), 2, "Planta (XY)"), ((0, 2), 1, "Frente (XZ)"), ((1, 2), 0, "Lateral (YZ)")]
    rows = [(PALETTE[labels] / 255, "clases"), (rgb, "color RGB")]

    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    for row, (colors, title) in enumerate(rows):
        for ax, ((i, j), depth, name) in zip(axes[row], views):
            order = sample[np.argsort(P[sample, depth] * (1 if depth == 2 else -1))]
            ax.scatter(P[order, i], P[order, j], c=colors[order], s=0.3, linewidths=0)
            ax.set_aspect("equal")
            ax.set_title(f"{name} - {title}", fontsize=13)
            ax.set_xlabel("xyz"[i])
            ax.set_ylabel("xyz"[j])

    present = np.unique(labels)
    handles = [Patch(color=PALETTE[c] / 255, label=CLASSES[c][0]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(present), fontsize=13, frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_png, dpi=70)
    plt.close(fig)


# Pipeline completo

def run(input_path, voxel=0.08, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    pcd, small = load_cloud(input_path, voxel)
    P, C, N = np.asarray(small.points), np.asarray(small.colors), np.asarray(small.normals)
    log(f"[0] {len(pcd.points):,} puntos -> {len(P):,} tras voxelizar a {voxel} m")

    f = local_features(P, N)
    eps8 = eps_from_kdistance(f["tree"], P, k=8)
    eps10 = eps_from_kdistance(f["tree"], P, k=10)
    log(f"[1] descriptores locales; eps por k-distancias: {eps8:.3f} m (k=8), {eps10:.3f} m (k=10)")

    labels = np.zeros(len(P), dtype=np.int64)
    history = []

    floor_z, z_top = stage_horizontal(P, f, labels, voxel)
    hf = P[:, 2] - floor_height_map(P, labels == PISO)
    history.append(("1. DBSCAN superficies horizontales", labels.copy()))
    log(f"[2] DBSCAN horizontales: piso z~{floor_z:.2f}, techo z~{z_top:.2f}")

    found = stage_columns(P, f, labels)
    history.append(("2. DBSCAN 2D columnas", labels.copy()))
    log(f"[3] DBSCAN 2D: {found.count(COLUMNA)} columnas, {found.count(MURO)} muros")

    people = stage_people(P, N, labels, hf, eps8)
    history.append(("3. DBSCAN personas y tanques", labels.copy()))
    n_people = sum(p["cls"] == PERSONA for p in people)
    n_tanks = sum(p["cls"] == TANQUE for p in people)
    log(f"[4] DBSCAN personas: {n_people} personas y {n_tanks} tanques "
        f"de {len(people)} candidatos con tamano de persona")

    ptype, km, names = stage_point_types(f, labels)
    log("[5] K-means tipos de punto: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))

    segs = stage_linear(P, N, f, labels, ptype, eps8)
    history.append(("4. K-means tipos + DBSCAN segmentos lineales", labels.copy()))
    count = lambda cls: sum(s["cls"] == cls for s in segs)
    log(f"[6] DBSCAN segmentos lineales: {count(VIGA)} vigas, {count(TUBERIA)} tuberias, "
        f"{count(POSTE)} postes")

    km_beams, km_pipes = stage_color(P, C, labels, segs, z_top)
    history.append(("5. K-means color de vigas y tuberias", labels.copy()))
    log(f"[7] K-means color: {count(VIGA_METAL)} vigas metalicas, {count(VIGA)} de concreto; "
        f"{count(TUBERIA_CLARA)} tuberias claras, {count(TUBERIA)} oscuras")

    objs = stage_objects(P, labels, hf, f["nvar"], z_top, eps10)
    history.append(("6. DBSCAN objetos", labels.copy()))
    log("[8] DBSCAN objetos: " + ", ".join(
        f"{sum(o['cls'] == c for o in objs)} {CLASSES[c][0]}"
        for c in (EQUIPO, VEHICULO, ESTRUCTURA, TECHO)))

    labels = smooth_labels(labels, f["nb"])
    history.append(("7. Suavizado por vecinos", labels.copy()))

    _, nearest = cKDTree(P).query(np.asarray(pcd.points), workers=-1)
    full = labels[nearest]
    log(f"[9] listo en {time.time() - t0:.1f}s")

    return dict(
        pcd=pcd, labels=full, P=P, C=C, N=N, vlabels=labels, f=f, hf=hf, history=history,
        eps8=eps8, eps10=eps10, people=people, ptype=ptype, km=km, names=names, segs=segs,
        km_beams=km_beams, km_pipes=km_pipes, objs=objs,
    )


# Programa principal

def main():
    ap = argparse.ArgumentParser(description="Segmentacion de una nube de puntos con clustering por etapas.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(HERE / "resultados" / "nube_segmentada.ply"))
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--show", action="store_true", help="abrir el visor 3D de Open3D al terminar")
    args = ap.parse_args()

    r = run(args.input, args.voxel)
    xyz, rgb = np.asarray(r["pcd"].points), np.asarray(r["pcd"].colors)

    out = Path(args.output)
    seg = save_results(r["pcd"], r["labels"], out)
    save_snapshots(xyz, r["labels"], rgb, out.with_suffix(".png"))

    print(f"\nGuardado: {out}")
    print(f"          {out.parent / 'clases'} (un PLY por clase)")
    print(f"          {out.with_suffix('.png')}\n")

    print(f"{'clase':22s}{'puntos':>12s}{'%':>8s}")
    for c, (name, _) in CLASSES.items():
        n = int(np.sum(r["labels"] == c))
        if n:
            print(f"{name:22s}{n:>12,}{100 * n / len(r['labels']):>7.1f}%")

    if args.show:
        o3d.visualization.draw_geometries([seg], window_name="Nube segmentada")


if __name__ == "__main__":
    main()
