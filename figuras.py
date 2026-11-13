import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import segmentacion_nube as S

OUT = S.HERE / "resultados" / "figuras"
GRIS = np.array([0.85, 0.85, 0.85])


# Utilidades de dibujo

def views(axes, P, colors, n=250_000, title=""):
    sample = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False)
    for ax, (i, j, depth, name) in zip(axes, ((0, 1, 2, "planta XY"), (0, 2, 1, "frente XZ"))):
        order = sample[np.argsort(P[sample, depth] * (1 if depth == 2 else -1))]
        ax.scatter(P[order, i], P[order, j], c=colors[order], s=0.25, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{title} ({name})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])


def class_colors(labels):
    colors = S.PALETTE[labels] / 255.0
    colors[labels == S.OTROS] = GRIS
    return colors


def legend(fig, labels):
    present = [c for c in np.unique(labels) if c != S.OTROS]
    handles = [Patch(color=S.PALETTE[c] / 255, label=S.CLASSES[c][0]) for c in present]
    handles.append(Patch(color=GRIS, label="sin clase todavia"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
               fontsize=10, frameon=False)


def save(fig, name, dpi=110):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=dpi)
    plt.close(fig)


def thumbnails(r, items, fname, title_of, half=0.9, cols=6):
    xyz, rgb = np.asarray(r["pcd"].points), np.asarray(r["pcd"].colors)
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.4 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, item in zip(axes.ravel(), items):
        x, y, z = item["center"]
        near = ((np.abs(xyz[:, 0] - x) < half) & (np.abs(xyz[:, 1] - y) < half)
                & (xyz[:, 2] > z - 1.3) & (xyz[:, 2] < z + 1.3))
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(xyz[near, 0], xyz[near, 2], c=rgb[near], s=0.6)
        ax.set_aspect("equal")
        text, color = title_of(item)
        ax.set_title(text, fontsize=8, color=color)

    save(fig, fname, dpi=70)


# Figura 1: grafico de k-distancias (eleccion de eps)

def fig_kdistance(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    P, tree = r["P"], r["f"]["tree"]

    for k, eps, color in ((8, r["eps8"], "C0"), (10, r["eps10"], "C1")):
        d, _ = tree.query(P[::10], k=k + 1, workers=-1)
        dk = np.sort(d[:, -1])
        ax.plot(np.linspace(0, 100, len(dk)), dk, color=color, label=f"distancia al {k}.o vecino")
        ax.axhline(eps, color=color, ls="--", lw=0.8)
        offset = 0.006 if k == 10 else -0.017
        ax.annotate(f"eps = {eps:.3f} m (percentil 95, k={k})", (2, eps + offset),
                    color=color, fontsize=9)

    ax.set_xlabel("puntos ordenados (%)")
    ax.set_ylabel("distancia (m)")
    ax.set_ylim(0, 0.4)
    ax.set_title("Grafico de k-distancias: el eps de DBSCAN se toma donde la curva se dispara")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85), fontsize=9)
    save(fig, "01_k_distancias.png")


# Figura 2: codo y silueta (eleccion de K)

def fig_elbow(r):
    f = r["f"]
    free = np.flatnonzero(r["history"][2][1] == S.OTROS)
    F = np.c_[f["lin"], f["pla"], f["sph"], f["e1z"], f["nz"], 3 * f["nvar"]][free]
    sample = np.random.default_rng(0).choice(len(F), 20_000, replace=False)

    Ks, inertia, silhouette = range(2, 11), [], []
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=5, random_state=0).fit(F)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(F[sample], km.labels_[sample]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))

    left.plot(list(Ks), inertia, "o-")
    left.axvline(6, color="0.5", ls="--")
    left.set_title("Codo: la ganancia cae a la mitad despues de K=6")
    left.set_xlabel("K")
    left.set_ylabel("inercia (WCSS)")

    right.plot(list(Ks), silhouette, "o-", color="C1")
    right.axvline(6, color="0.5", ls="--")
    right.set_ylim(0, 0.5)
    right.set_title("Silueta: casi plana, no distingue un K")
    right.set_xlabel("K")
    right.set_ylabel("silueta media")

    save(fig, "02_kmeans_codo_silueta.png")


# Figura 3: la nube despues de cada etapa

def fig_stages(r):
    for i, (title, labels) in enumerate(r["history"]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        views(axes, r["P"], class_colors(labels), title=title)
        legend(fig, labels)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(OUT / f"03_etapa_{i + 1}.png", dpi=80)
        plt.close(fig)


# Figura 4: tipos de punto encontrados por K-means

def fig_point_types(r):
    names = sorted(set(r["names"]))
    palette = dict(zip(names, plt.cm.tab10(np.arange(len(names)))[:, :3]))

    colors = np.tile(GRIS, (len(r["P"]), 1))
    for name in names:
        colors[r["ptype"] == name] = palette[name]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    views(axes, r["P"], colors, title="K-means: tipo de punto")
    handles = [Patch(color=palette[name], label=name) for name in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "04_kmeans_tipos.png", dpi=80)
    plt.close(fig)


# Figura 5: viga o tuberia segun la forma de la seccion

def fig_beam_pipe(r):
    fig, ax = plt.subplots(figsize=(7, 5))

    for cls in (S.VIGA, S.VIGA_METAL, S.TUBERIA, S.TUBERIA_CLARA, S.POSTE):
        group = [s for s in r["segs"] if s["cls"] == cls]
        ax.scatter([s["rel"] for s in group], [s["radial"] for s in group],
                   s=[s["n"] / 15 for s in group], color=S.PALETTE[cls] / 255,
                   alpha=0.7, label=S.CLASSES[cls][0])

    ax.axvline(0.12, color="0.4", ls="--", lw=0.8)
    ax.axvline(0.18, color="0.4", ls=":", lw=0.8)
    ax.axhline(0.9, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel("error del ajuste de circulo / radio  (bajo = seccion circular)")
    ax.set_ylabel("alineacion normal-radio  (1 = las normales apuntan al eje)")
    ax.set_title("Cada segmento lineal: tuberia si su seccion es un circulo")
    ax.legend(fontsize=9)
    save(fig, "05_vigas_vs_tuberias.png")


# Figura 6: K-means sobre el color de vigas y tuberias

def fig_color(r, classes, km, xlabel, low, high, what, fname):
    group = [s for s in r["segs"] if s["cls"] in classes]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for s in group:
        ax.scatter(s["color_value"] * 255, s["z"], s=s["n"] / 6, color=s["rgb"],
                   edgecolors="k", linewidths=0.4)

    centers = km.cluster_centers_[:, 0] * 255
    for c in centers:
        ax.axvline(c, color="0.5", ls=":")

    border = centers.mean()
    top = ax.get_ylim()[1] - 0.8
    ax.axvline(border, color="k", ls="--")
    ax.text(border + 2, top, f"{high} ->", fontsize=9)
    ax.text(border - 2, top, f"<- {low}", fontsize=9, ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("altura z (m)")
    ax.set_title(f"K-means (K=2) sobre el color de cada {what} (cada punto con su color real)")
    save(fig, fname)


# Figuras 7 y 8: personas, tanques y vehiculos

def fig_people(r):
    people = sorted(r["people"], key=lambda p: p["ring"])
    names = {
        S.PERSONA: ("PERSONA", "green"),
        S.TANQUE: ("TANQUE", "teal"),
        S.OTROS: ("rechazado", "red"),
    }

    def title(p):
        label, color = names[p["cls"]]
        x, y, _ = p["center"]
        text = (f"{label} ({x:.1f}, {y:.1f})\n"
                f"anillo {p['ring']:.2f}, cilindro R={p['R']:.2f} alin={p['align']:.2f}")
        return text, color

    thumbnails(r, people, "07_personas.png", title)


def fig_vehicles(r):
    cars = [o for o in r["objs"] if o["cls"] == S.VEHICULO]
    if not cars:
        return

    def title(o):
        x, y, _ = o["center"]
        text = (f"vehiculo ({x:.1f}, {y:.1f})\n"
                f"{o['length']:.1f} x {o['width']:.1f} x {o['height']:.1f} m")
        return text, "black"

    thumbnails(r, cars, "08_vehiculos.png", title, half=3.5, cols=len(cars))


# Programa principal

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = S.run(S.DEFAULT_INPUT)

    fig_kdistance(r)
    fig_elbow(r)
    fig_stages(r)
    fig_point_types(r)
    fig_beam_pipe(r)
    fig_color(r, (S.VIGA, S.VIGA_METAL), r["km_beams"], "croma = max(R,G,B) - min(R,G,B)",
              "viga de concreto", "viga metalica (pintada)", "viga", "06_color_vigas.png")
    fig_color(r, (S.TUBERIA, S.TUBERIA_CLARA), r["km_pipes"], "brillo = max(R,G,B)",
              "tuberia oscura", "tuberia clara", "tuberia", "06_color_tuberias.png")
    fig_people(r)
    fig_vehicles(r)

    print(f"Figuras guardadas en {OUT}")


if __name__ == "__main__":
    main()
