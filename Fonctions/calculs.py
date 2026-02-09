import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri


# ============================================================
# 1) Utils géométrie


def ensure_closed_polygon(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 3:
        raise ValueError("Polygone invalide: moins de 3 points.")
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def arc_points_from_center(xstart, ystart, xend, yend, xcenter, ycenter, sens, n=120):
    a1 = math.atan2(ystart - ycenter, xstart - xcenter)
    a2 = math.atan2(yend - ycenter, xend - xcenter)

    r1 = math.hypot(xstart - xcenter, ystart - ycenter)
    r2 = math.hypot(xend - xcenter, yend - ycenter)
    if not np.isclose(r1, r2, rtol=1e-6, atol=1e-9):
        print(f"[WARN] Rayon start/end différent arc (r1={r1}, r2={r2}). On utilise r1.")

    sens_norm = (sens or "").strip().lower()
    cw = sens_norm in ["horaire", "cw", "clockwise"]
    ccw = sens_norm in ["trigo", "anti-horaire", "antihoraire", "ccw", "counterclockwise"]
    if not (cw or ccw):
        ccw = True

    if ccw:
        if a2 <= a1:
            a2 += 2 * math.pi
        angles = np.linspace(a1, a2, n)
    else:
        if a2 >= a1:
            a2 -= 2 * math.pi
        angles = np.linspace(a1, a2, n)

    xs = xcenter + np.cos(angles) * r1
    ys = ycenter + np.sin(angles) * r1
    return np.column_stack([xs, ys])


def point_segment_distance(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)
    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return math.hypot(px - bx, py - by)


def polyline_segments(points: np.ndarray):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        yield (float(x1), float(y1), float(x2), float(y2))


# ============================================================
# 2) JSON extraction

def load_json(filepath="projetTest.json") -> dict:
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Fichier introuvable: {fp.resolve()}")
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_shapes(data: dict) -> dict:
    return {s["id"]: s for s in data.get("shapes", [])}


def get_domain_polygon(data: dict, domain_name="Domaine_1") -> np.ndarray:
    shapes_by_id = index_shapes(data)
    edits = data["domainEdits"].get(domain_name, [])
    if not edits:
        raise ValueError(f"Aucun domainEdit pour {domain_name}")

    for ed in edits:
        sid = ed.get("shapeId")
        sh = shapes_by_id.get(sid)
        if sh and sh.get("kind") == "POLY":
            pts = np.array(sh["polyPoints"], dtype=float)
            return ensure_closed_polygon(pts)

    raise ValueError("Ce solveur attend un POLY pour définir le domaine.")


def get_mesh_steps(data: dict, domain_name="Domaine_1"):
    spec = data["meshSpecs"].get(domain_name)
    if not spec:
        raise ValueError(f"Aucun meshSpecs pour {domain_name}")
    kind = spec.get("kind", "TRI_REG")
    dx = float(spec.get("dx", 1.0))
    dy = float(spec.get("dy", 1.0))
    return kind, dx, dy


def infer_length_scale(poly_pts: np.ndarray) -> float:
    """
    Heuristique: si la taille du domaine ~ centaines -> on suppose "mm" et on convertit en m.
    Sinon on suppose déjà en mètres.
    """
    span = float(np.max(poly_pts) - np.min(poly_pts))
    if 20.0 <= span <= 50000.0:
        return 1e-3  # mm -> m
    return 1.0


# ============================================================
# 3) Mesh grille + mask + triangulation

def build_grid_and_mask(poly_pts: np.ndarray, dx: float, dy: float, padding=0.0):
    xmin, ymin = poly_pts.min(axis=0)
    xmax, ymax = poly_pts.max(axis=0)
    xmin -= padding; ymin -= padding; xmax += padding; ymax += padding

    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dy, dy)
    X, Y = np.meshgrid(x, y)

    points = np.column_stack([X.ravel(), Y.ravel()])
    path = MplPath(poly_pts)
    inside = path.contains_points(points, radius=1e-9)
    mask = inside.reshape(X.shape)

    return X, Y, mask, (xmin, xmax, ymin, ymax), path


def build_triangulation_from_grid(X: np.ndarray, Y: np.ndarray, path: MplPath):
    px = X.ravel()
    py = Y.ravel()
    tri = mtri.Triangulation(px, py)

    tri_pts = np.stack([px[tri.triangles], py[tri.triangles]], axis=-1)
    centroids = tri_pts.mean(axis=1)
    tri_inside = path.contains_points(centroids)

    tri.set_mask(~tri_inside)
    return tri


def plot_mesh(poly_pts, X, Y, mask, tri, extent):
    xmin, xmax, ymin, ymax = extent
    plt.figure(figsize=(7, 6))
    plt.imshow(mask.astype(int), origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="equal", interpolation="nearest")
    plt.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2)
    plt.title("Domaine mask + contour")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()

    plt.figure(figsize=(7, 6))
    plt.triplot(tri, linewidth=0.6)
    plt.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2)
    plt.title("TRI_REG (triangles masqués hors domaine)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Parsing BC/IC (Dirichlet / Neumann flux / Open / Sym)

def parse_bc_value(s: str):
    if s is None:
        return ("None", None)
    if isinstance(s, (int, float)):
        return ("Const", float(s))
    s = str(s).strip()
    if s.lower() == "nan":
        return ("NaN", None)
    if "," in s:
        a, b = s.split(",", 1)
        a = a.strip()
        b = b.strip()
        try:
            bv = float(b)
        except ValueError:
            bv = b
        return (a, bv)
    try:
        return ("Const", float(s))
    except ValueError:
        return ("Raw", s)


def parse_mass_tokens(mass_str: str, nspecies: int):
    """
    Supporte:
      - Dirichlet: "0.17,0.83,0,0,0,0"
      - OpenBoundary / AxialSymetry / Adiabatique (Neumann 0)
      - FluxImpose:val (recommandé) par espèce
    Retour:
      dirichlet_vec: array or None
      flux_vec: array or None  (flux ENTRANT vers le domaine)
      mode_vec: list[str] taille nspecies : "DIR"/"FLUX"/"OPEN"/"SYM"/"NEU0"
    """
    mode = ["OPEN"] * nspecies
    dirv = np.zeros(nspecies, dtype=float)
    fluxv = np.zeros(nspecies, dtype=float)

    if mass_str is None:
        return None, None, mode

    parts = [p.strip() for p in str(mass_str).split(",")]
    parts = (parts + ["OpenBoundary"] * nspecies)[:nspecies]

    has_dir = False
    has_flux = False

    for i, tok in enumerate(parts):
        t = tok.strip()
        tl = t.lower()

        if tl.startswith("openboundary"):
            mode[i] = "OPEN"
        elif tl.startswith("axialsymetry"):
            mode[i] = "SYM"
        elif tl.startswith("adiabatique") or tl.startswith("isolant"):
            mode[i] = "NEU0"  # flux nul
        elif tl.startswith("fluximpose"):
            # on attend surtout "FluxImpose:val"
            val = 0.0
            if ":" in t:
                _, v = t.split(":", 1)
                try:
                    val = float(v)
                except Exception:
                    val = 0.0
            elif "=" in t:
                _, v = t.split("=", 1)
                try:
                    val = float(v)
                except Exception:
                    val = 0.0
            else:
                # si "FluxImpose,100" est présent, parse_bc_value ne s'applique pas ici car mass est CSV.
                # Donc on laisse 0 (ou tu imposes la forme FluxImpose:100).
                val = 0.0

            fluxv[i] = val  # flux ENTRANT
            mode[i] = "FLUX"
            has_flux = True
        else:
            # Dirichlet float
            try:
                dirv[i] = float(t)
                mode[i] = "DIR"
                has_dir = True
            except ValueError:
                mode[i] = "OPEN"

    dirichlet_vec = dirv if has_dir else None
    flux_vec = fluxv if has_flux else None
    return dirichlet_vec, flux_vec, mode


# ============================================================
# 5) Détection frontières (Ligne et arc)

def boundary_nodes_from_shapes(data, X, Y, mask, tol_factor=1.5):
    shapes_by_id = index_shapes(data)

    dxg = float(np.nanmean(np.diff(X[0, :]))) if X.shape[1] > 1 else 1.0
    dyg = float(np.nanmean(np.diff(Y[:, 0]))) if Y.shape[0] > 1 else 1.0
    tol = tol_factor * min(dxg, dyg)

    Ny, Nx = mask.shape

    # 1) boundary_layer = inside avec un voisin outside
    boundary_layer = []
    for j in range(Ny):
        for i in range(Nx):
            if not mask[j, i]:
                continue
            outside_nb = False
            for (jj, ii) in [(j+1,i),(j-1,i),(j,i+1),(j,i-1)]:
                if 0 <= jj < Ny and 0 <= ii < Nx:
                    if not mask[jj, ii]:
                        outside_nb = True
                        break
                else:
                    outside_nb = True
                    break
            if outside_nb:
                boundary_layer.append((j, i))

    # 2) segments par frontière
    boundary_segs = {}
    for bname, shape_ids in data.get("boundaryLines", {}).items():
        segs = []
        for sid in shape_ids:
            sh = shapes_by_id.get(sid)
            if not sh:
                continue
            if sh.get("kind") == "LINE":
                segs.append((float(sh["x1"]), float(sh["y1"]), float(sh["x2"]), float(sh["y2"])))
            elif sh.get("kind") == "ARC":
                arc_pts = arc_points_from_center(
                    float(sh["xstart"]), float(sh["ystart"]),
                    float(sh["xend"]), float(sh["yend"]),
                    float(sh["xcentre"]), float(sh["ycentre"]),
                    sh.get("sens", "trigo"),
                    n=300
                )
                for x1, y1, x2, y2 in polyline_segments(arc_pts):
                    segs.append((x1, y1, x2, y2))
        boundary_segs[bname] = segs

    out = {bname: [] for bname in boundary_segs.keys()}

    # 3) assignation : NE PAS jeter si best_d > tol
    for (j, i) in boundary_layer:
        px = float(X[j, i]); py = float(Y[j, i])

        best_name = None
        best_d = float("inf")

        for bname, segs in boundary_segs.items():
            if not segs:
                continue
            dmin = float("inf")
            for x1, y1, x2, y2 in segs:
                d = point_segment_distance(px, py, x1, y1, x2, y2)
                if d < dmin:
                    dmin = d
            if dmin < best_d:
                best_d = dmin
                best_name = bname

        if best_name is not None:
            out[best_name].append((j, i))

    # nettoyage + log
    for bname in out:
        out[bname] = list(set(out[bname]))
        print(f"[BC] {bname}: {len(out[bname])} noeuds (tol~{tol:g}, assign=nearest)")

    return out




# ============================================================
# 6) Opérateurs numériques (div, grad, convection upwind, solve SOR)

def compute_divergence(ux, uy, mask, dx, dy):
    Ny, Nx = ux.shape
    div = np.zeros_like(ux, dtype=float)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            if not mask[j, i]:
                continue

            # dudx : central si possible, sinon one-sided si un seul voisin existe
            if mask[j, i+1] and mask[j, i-1]:
                dudx = (ux[j, i+1] - ux[j, i-1]) / (2.0 * dx)
            elif mask[j, i+1] and not mask[j, i-1]:
                dudx = (ux[j, i+1] - ux[j, i]) / dx
            elif mask[j, i-1] and not mask[j, i+1]:
                dudx = (ux[j, i] - ux[j, i-1]) / dx
            else:
                dudx = 0.0

            # dvdy : central si possible, sinon one-sided
            if mask[j+1, i] and mask[j-1, i]:
                dvdy = (uy[j+1, i] - uy[j-1, i]) / (2.0 * dy)
            elif mask[j+1, i] and not mask[j-1, i]:
                dvdy = (uy[j+1, i] - uy[j, i]) / dy
            elif mask[j-1, i] and not mask[j+1, i]:
                dvdy = (uy[j, i] - uy[j-1, i]) / dy
            else:
                dvdy = 0.0

            div[j, i] = dudx + dvdy

    return div



def grad_scalar(phi, mask, dx, dy):
    Ny, Nx = phi.shape
    gx = np.zeros_like(phi, dtype=float)
    gy = np.zeros_like(phi, dtype=float)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            if not mask[j, i]:
                continue

            # dphi/dx
            if mask[j, i+1] and mask[j, i-1]:
                gx[j, i] = (phi[j, i+1] - phi[j, i-1]) / (2.0 * dx)
            elif mask[j, i+1] and not mask[j, i-1]:
                gx[j, i] = (phi[j, i+1] - phi[j, i]) / dx
            elif mask[j, i-1] and not mask[j, i+1]:
                gx[j, i] = (phi[j, i] - phi[j, i-1]) / dx
            else:
                gx[j, i] = 0.0

            # dphi/dy
            if mask[j+1, i] and mask[j-1, i]:
                gy[j, i] = (phi[j+1, i] - phi[j-1, i]) / (2.0 * dy)
            elif mask[j+1, i] and not mask[j-1, i]:
                gy[j, i] = (phi[j+1, i] - phi[j, i]) / dy
            elif mask[j-1, i] and not mask[j+1, i]:
                gy[j, i] = (phi[j, i] - phi[j-1, i]) / dy
            else:
                gy[j, i] = 0.0

    return gx, gy



def upwind_convection(phi, ux, uy, mask, dx, dy):
    Ny, Nx = phi.shape
    conv = np.zeros_like(phi, dtype=float)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            if not mask[j, i]:
                continue

            u = float(ux[j, i])
            v = float(uy[j, i])

            # Upwind en x avec fallback one-sided si un seul voisin existe
            if u >= 0:
                if mask[j, i-1]:
                    dphidx = (phi[j, i] - phi[j, i-1]) / dx
                elif mask[j, i+1]:
                    dphidx = (phi[j, i+1] - phi[j, i]) / dx
                else:
                    dphidx = 0.0
            else:
                if mask[j, i+1]:
                    dphidx = (phi[j, i+1] - phi[j, i]) / dx
                elif mask[j, i-1]:
                    dphidx = (phi[j, i] - phi[j, i-1]) / dx
                else:
                    dphidx = 0.0

            # Upwind en y
            if v >= 0:
                if mask[j-1, i]:
                    dphidy = (phi[j, i] - phi[j-1, i]) / dy
                elif mask[j+1, i]:
                    dphidy = (phi[j+1, i] - phi[j, i]) / dy
                else:
                    dphidy = 0.0
            else:
                if mask[j+1, i]:
                    dphidy = (phi[j+1, i] - phi[j, i]) / dy
                elif mask[j-1, i]:
                    dphidy = (phi[j, i] - phi[j-1, i]) / dy
                else:
                    dphidy = 0.0

            # terme RHS pour dphi/dt = ...  => -(u·grad phi)
            conv[j, i] = -(u * dphidx + v * dphidy)

    return conv



def poisson_sor(phi, rhs, mask, dx, dy,
                fixed_mask=None, fixed_value=0.0,
                fixed_value_array=None,
                n_iter=1500, omega=1.5,
                tol_inner=1e-8,
                check_every=20):
    """
    Solve Laplace(phi) = rhs avec SOR sur grille masquée.
    Ajout: arrêt anticipé si max_update < tol_inner.
    """
    Ny, Nx = phi.shape
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)

    if fixed_mask is None:
        fixed_mask = np.zeros_like(mask, dtype=bool)

    # gauge si aucun Dirichlet
    if not fixed_mask.any():
        js, is_ = np.argwhere(mask)[0]
        fixed_mask[js, is_] = True
        if fixed_value_array is None:
            phi[js, is_] = fixed_value
        else:
            phi[js, is_] = fixed_value_array[js, is_]

    # pré-impose
    if fixed_value_array is not None:
        phi[fixed_mask] = fixed_value_array[fixed_mask]
    else:
        phi[fixed_mask] = fixed_value

    for it in range(1, n_iter + 1):
        max_update = 0.0

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if not mask[j, i] or fixed_mask[j, i]:
                    continue

                pe = phi[j, i+1] if mask[j, i+1] else phi[j, i]
                pw = phi[j, i-1] if mask[j, i-1] else phi[j, i]
                pn = phi[j+1, i] if mask[j+1, i] else phi[j, i]
                ps = phi[j-1, i] if mask[j-1, i] else phi[j, i]

                newv = ((pe + pw)/dx2 + (pn + ps)/dy2 - rhs[j, i]) / denom

                oldv = phi[j, i]
                phi[j, i] = oldv + omega * (newv - oldv)
                du = abs(phi[j, i] - oldv)
                if du > max_update:
                    max_update = du

        # re-impose
        if fixed_value_array is not None:
            phi[fixed_mask] = fixed_value_array[fixed_mask]
        else:
            phi[fixed_mask] = fixed_value

        if (it % check_every) == 0 and max_update < tol_inner:
            break

    return phi



# ============================================================
# 7) Diffusion implicite avec flux Neumann (T et espèces)

def build_flux_faces_for_nodes(nodes, mask):
    Ny, Nx = mask.shape
    faces = {}
    for (j, i) in nodes:
        if not mask[j, i]:
            continue
        if i+1 < Nx and not mask[j, i+1]:
            faces[(j, i, "E")] = True
        if i-1 >= 0 and not mask[j, i-1]:
            faces[(j, i, "W")] = True
        if j+1 < Ny and not mask[j+1, i]:
            faces[(j, i, "N")] = True
        if j-1 >= 0 and not mask[j-1, i]:
            faces[(j, i, "S")] = True
    return faces


def diffusion_implicit_step_with_flux(
    phi, rhs_explicit, mask, dx, dy, alpha, dt,
    dirichlet_mask=None, dirichlet_value=None,
    neumann_flux_faces=None,
    K_for_flux=None,
    n_iter=600, omega=1.25,
    tol_inner=1e-8,
    check_every=20
):
    """
    Solve: (phi^{n+1} - phi^n)/dt = alpha * Laplace(phi^{n+1}) + rhs_explicit
    via SOR, avec Neumann par ghost-cell.

    Ajout: arrêt anticipé si la variation max devient < tol_inner.
    """
    Ny, Nx = phi.shape
    dx2 = dx * dx
    dy2 = dy * dy
    a = dt * alpha
    diag = 1.0 + 2.0*a*(1.0/dx2 + 1.0/dy2)

    if dirichlet_mask is None:
        dirichlet_mask = np.zeros_like(mask, dtype=bool)

    b = phi + dt * rhs_explicit

    if neumann_flux_faces is None:
        neumann_flux_faces = {}
    K = float(K_for_flux) if K_for_flux is not None else 1.0
    K = max(K, 1e-30)

    # pré-impose Dirichlet (utile pour stabiliser)
    if dirichlet_value is not None:
        phi[dirichlet_mask] = dirichlet_value[dirichlet_mask]

    for it in range(1, n_iter + 1):
        max_update = 0.0

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if not mask[j, i] or dirichlet_mask[j, i]:
                    continue

                # voisins / ghost Neumann
                if mask[j, i+1]:
                    pe = phi[j, i+1]
                else:
                    q = neumann_flux_faces.get((j, i, "E"), 0.0)
                    pe = phi[j, i] - dx * q / K

                if mask[j, i-1]:
                    pw = phi[j, i-1]
                else:
                    q = neumann_flux_faces.get((j, i, "W"), 0.0)
                    pw = phi[j, i] - dx * q / K

                if mask[j+1, i]:
                    pn = phi[j+1, i]
                else:
                    q = neumann_flux_faces.get((j, i, "N"), 0.0)
                    pn = phi[j, i] - dy * q / K

                if mask[j-1, i]:
                    ps = phi[j-1, i]
                else:
                    q = neumann_flux_faces.get((j, i, "S"), 0.0)
                    ps = phi[j, i] - dy * q / K

                nb = a * ((pe + pw)/dx2 + (pn + ps)/dy2)
                newv = (b[j, i] + nb) / diag

                oldv = phi[j, i]
                phi[j, i] = oldv + omega * (newv - oldv)
                du = abs(phi[j, i] - oldv)
                if du > max_update:
                    max_update = du

        # re-impose Dirichlet à chaque sweep
        if dirichlet_value is not None:
            phi[dirichlet_mask] = dirichlet_value[dirichlet_mask]

        # arrêt anticipé (pas à chaque sweep pour éviter overhead)
        if (it % check_every) == 0 and max_update < tol_inner:
            break

    return phi



def sanitize_inplace(arr, mask, clip_abs=1e12):
    """
    Evite explosion silencieuse: remplace nan/inf, clip amplitude.
    """
    a = arr
    bad = ~np.isfinite(a)
    if np.any(bad & mask):
        a[bad & mask] = 0.0
    a[mask] = np.clip(a[mask], -clip_abs, clip_abs)


# ============================================================
# 8) Modèles physiques

R_GAS = 8.314462618
MEV_TO_J = 1.602176634e-13

SPECIES = ["Li", "Pb", "nfast", "nslow", "He4", "Tritium"]
S_INDEX = {name: i for i, name in enumerate(SPECIES)}


def eval_material_coeffs(data):
    mv = data.get("matVars", {}).get("LiPb", {})

    def get_float(name, default):
        expr = str(mv.get(name, {}).get("funcExpr", "")).strip()
        try:
            return float(expr)
        except Exception:
            return float(default)

    mu = get_float("mu", 1.7e-3)
    k = get_float("k", 35.0)
    cp = get_float("cp", 945.0)
    sigma_e = get_float("sigma_e", 4e6)

    rho = 9000.0
    return rho, mu, k, cp, sigma_e


def reaction_rate_arrhenius(A, B, T):
    return A * np.exp(B / (R_GAS * np.maximum(T, 1.0)))


def parse_reactions(data):
    rx = data.get("reactions", {}).get("LiPb", [])
    out = []
    for r in rx:
        reactants = [s.strip() for s in str(r.get("reactants", "")).split(",") if s.strip()]
        products = [s.strip() for s in str(r.get("products", "")).split(",") if s.strip()]
        energy = float(r.get("energyMeV", 0.0))

        if "A" in r and "B" in r:
            model = "ARRHENIUS"
            A = float(r["A"])
            B = float(r["B"])
            k0 = None
        else:
            model = "CONST"
            try:
                k0 = float(r.get("kExpr", 0.0))
            except Exception:
                k0 = 0.0
            A = None
            B = None

        out.append({
            "model": model,
            "k0": k0,
            "A": A,
            "B": B,
            "reactants": reactants,
            "products": products,
            "Er_J": energy * MEV_TO_J
        })
    return out


def compute_reaction_sources(C, T, reactions):
    ns, Ny, Nx = C.shape
    S_C = np.zeros_like(C)
    Q = np.zeros((Ny, Nx), dtype=float)

    for r in reactions:
        if r["model"] == "CONST":
            k = float(r["k0"])
        else:
            k = reaction_rate_arrhenius(r["A"], r["B"], T)

        rate = np.ones((Ny, Nx), dtype=float)
        for sp in r["reactants"]:
            if sp not in S_INDEX:
                continue
            rate *= np.maximum(C[S_INDEX[sp]], 0.0)
        rate *= k

        for sp in r["reactants"]:
            if sp in S_INDEX:
                S_C[S_INDEX[sp]] -= rate
        for sp in r["products"]:
            if sp in S_INDEX:
                S_C[S_INDEX[sp]] += rate

        Q += rate * r["Er_J"]

    return S_C, Q


def get_imposed_B_from_json(data):
    bc = data.get("boundarySpecs", {})
    for _, spec in bc.items():
        btype, bval = parse_bc_value(spec.get("B", None))
        if btype in ["FonctionImposee", "Const"] and bval is not None:
            return float(bval)
    return 5.0


def mhd_lorentz_force(ux, uy, Bz, sigma_e, mask):
    fx = np.zeros_like(ux)
    fy = np.zeros_like(uy)
    coef = -sigma_e * (Bz ** 2)
    fx[mask] = coef * ux[mask]
    fy[mask] = coef * uy[mask]
    return fx, fy


# ============================================================
# 9) Champs
# ============================================================

@dataclass
class Fields:
    ux: np.ndarray
    uy: np.ndarray
    p: np.ndarray
    T: np.ndarray
    C: np.ndarray


def build_initial_fields(data, mask, nspecies=6):
    Ny, Nx = mask.shape
    ux = np.zeros((Ny, Nx), dtype=float)
    uy = np.zeros((Ny, Nx), dtype=float)
    p = np.zeros((Ny, Nx), dtype=float)
    T = np.zeros((Ny, Nx), dtype=float)
    C = np.zeros((nspecies, Ny, Nx), dtype=float)

    init_specs = data.get("initSpecs", {})
    ini = init_specs.get("CI_1", None) or (list(init_specs.values())[0] if init_specs else None)

    if ini is None:
        T[mask] = 350.0
        return Fields(ux, uy, p, T, C)

    ux0 = float(ini.get("ux", 0.0))
    uy0 = float(ini.get("uy", 0.0))
    p0 = float(ini.get("p", 0.0))
    T0 = float(ini.get("T", 350.0))
    ux[mask] = ux0
    uy[mask] = uy0
    p[mask] = p0
    T[mask] = T0

    m = str(ini.get("mass", "")).strip()
    if m:
        parts = [p.strip() for p in m.split(",")]
        parts = (parts + ["0"] * nspecies)[:nspecies]
        for s in range(nspecies):
            try:
                C[s][mask] = float(parts[s])
            except Exception:
                C[s][mask] = 0.0

    return Fields(ux, uy, p, T, C)


# ============================================================
# 10) Application BC -> Dirichlet + flux Neumann

def apply_bcs_build_masks_and_fluxes(data, fields, bnodes, mask):
    bc_specs = data.get("boundarySpecs", {})
    nspecies = fields.C.shape[0]
    Ny, Nx = mask.shape

    # ----- Dirichlet u -----
    dir_u = np.zeros((Ny, Nx), dtype=bool)
    dir_ux_val = np.zeros((Ny, Nx), dtype=float)
    dir_uy_val = np.zeros((Ny, Nx), dtype=float)

    # ----- Dirichlet p -----
    dir_p = np.zeros((Ny, Nx), dtype=bool)
    dir_p_val = np.zeros((Ny, Nx), dtype=float)

    # ----- Dirichlet T -----
    dir_T = np.zeros((Ny, Nx), dtype=bool)
    dir_T_val = np.zeros((Ny, Nx), dtype=float)

    # ----- Dirichlet C -----
    dir_C = np.zeros((nspecies, Ny, Nx), dtype=bool)
    dir_C_val = np.zeros((nspecies, Ny, Nx), dtype=float)

    # ----- Neumann flux faces -----
    flux_T_faces = {}
    flux_C_faces = [dict() for _ in range(nspecies)]

    # Pour gérer OpenBoundary/Axisym pour la vitesse : du/dn = 0
    flux_ux_faces = {}
    flux_uy_faces = {}

    for bname, nodes in bnodes.items():
        spec = bc_specs.get(bname, {})

        ux_type, ux_val = parse_bc_value(spec.get("ux", None))
        uy_type, uy_val = parse_bc_value(spec.get("uy", None))
        p_type,  p_val  = parse_bc_value(spec.get("p",  None))
        T_type,  T_val  = parse_bc_value(spec.get("T",  None))

        mass_dir, mass_flux, mass_mode = parse_mass_tokens(spec.get("mass", None), nspecies)

        # =========================
        # A) Pression : Dirichlet si FonctionImposee/Const
        # =========================
        for (j, i) in nodes:
            if not mask[j, i]:
                continue
            if p_type in ["FonctionImposee", "Const"]:
                dir_p[j, i] = True
                dir_p_val[j, i] = float(p_val)

        # =========================
        # B) Vitesse
        #  - NoSlip / FonctionVitesseImposee / Const => Dirichlet
        #  - Axisym (x=0) : impose ux=0 ; impose d(uy)/dn = 0 via Neumann0
        #  - OpenBoundary : impose du/dn = 0 via Neumann0 (sur ux et uy)
        # =========================
        is_open_u = (ux_type == "OpenBoundary") or (uy_type == "OpenBoundary")
        is_sym_u  = (ux_type == "AxialSymetry") or (uy_type == "AxialSymetry")

        # Neumann0 pour u sur ces frontières
        if is_open_u or is_sym_u:
            faces = build_flux_faces_for_nodes(nodes, mask)
            for key in faces.keys():
                flux_ux_faces[key] = 0.0
                flux_uy_faces[key] = 0.0

        for (j, i) in nodes:
            if not mask[j, i]:
                continue

            # Axisym sur x=0 : ux=0 (impermeable). On ne fixe pas uy.
            if is_sym_u:
                dir_u[j, i] = True
                dir_ux_val[j, i] = 0.0
                # ne pas imposer uy ici
                continue

            # Dirichlet classiques
            if ux_type in ["NoSlip", "FonctionVitesseImposee", "Const"]:
                dir_u[j, i] = True
                dir_ux_val[j, i] = 0.0 if ux_type == "NoSlip" else float(ux_val or 0.0)

            if uy_type in ["NoSlip", "FonctionVitesseImposee", "Const"]:
                dir_u[j, i] = True
                dir_uy_val[j, i] = 0.0 if uy_type == "NoSlip" else float(uy_val or 0.0)

        # =========================
        # C) Température
        #  - Dirichlet si FonctionTempIposee / Const
        #  - FluxImpose,val => Neumann (flux entrant)
        #  - Adiabatique/OpenBoundary/Axisym/Isolant => Neumann 0 (rien à faire)
        # =========================
        qT_out = 0.0
        if isinstance(T_type, str) and T_type.lower() in ["fluximpose", "heatflux"]:
            q_in = float(T_val)         # entrant vers le domaine
            qT_out = -q_in              # convention outward

        for (j, i) in nodes:
            if not mask[j, i]:
                continue
            if T_type in ["FonctionTempIposee", "Const"]:
                dir_T[j, i] = True
                dir_T_val[j, i] = float(T_val)

        if abs(qT_out) > 0.0:
            faces = build_flux_faces_for_nodes(nodes, mask)
            for key in faces.keys():
                flux_T_faces[key] = qT_out

        # =========================
        # D) Espèces
        #  - Dirichlet si valeurs numériques
        #  - FluxImpose:val => Neumann flux espèce (entrant)
        #  - Adiabatique/OpenBoundary/Axisym => Neumann 0 (rien à faire)
        # =========================
        if mass_dir is not None:
            for (j, i) in nodes:
                if not mask[j, i]:
                    continue
                for s in range(nspecies):
                    if mass_mode[s] == "DIR":
                        dir_C[s, j, i] = True
                        dir_C_val[s, j, i] = float(mass_dir[s])

        if mass_flux is not None:
            faces = build_flux_faces_for_nodes(nodes, mask)
            for s in range(nspecies):
                if mass_mode[s] != "FLUX":
                    continue
                F_in = float(mass_flux[s])
                F_out = -F_in
                for key in faces.keys():
                    flux_C_faces[s][key] = F_out

    # Applique Dirichlet sur champs
    fields.ux[dir_u] = dir_ux_val[dir_u]
    fields.uy[dir_u] = dir_uy_val[dir_u]
    fields.p[dir_p]  = dir_p_val[dir_p]
    fields.T[dir_T]  = dir_T_val[dir_T]
    for s in range(nspecies):
        fields.C[s][dir_C[s]] = dir_C_val[s][dir_C[s]]

    return (
        dir_u, dir_ux_val, dir_uy_val,
        dir_p, dir_p_val,
        dir_T, dir_T_val,
        dir_C, dir_C_val,
        flux_ux_faces, flux_uy_faces,
        flux_T_faces,
        flux_C_faces
    )



# ============================================================
# 11) Solveur couplé pseudo-temps

def run_coupled_solver(data, X, Y, mask, poly_pts, tri, extent):
    domain_name = "Domaine_1"
    _, dx_raw, dy_raw = get_mesh_steps(data, domain_name=domain_name)

    # ----- IMPORTANT: conversion unités géométriques -> mètres -----
    scale = infer_length_scale(poly_pts)
    dx = dx_raw * scale
    dy = dy_raw * scale
    print(f"[UNIT] length_scale={scale:g} (dx={dx_raw} -> {dx} m)")

    nspecies = len(SPECIES)
    fields = build_initial_fields(data, mask, nspecies=nspecies)

    bnodes = boundary_nodes_from_shapes(data, X, Y, mask, tol_factor=0.6)

    solver = data.get("solver", {})
    tol = float(solver.get("tol", 1e-6))
    itmax = int(solver.get("iters", 10000))

    rho, mu, k, cp, sigma_e = eval_material_coeffs(data)
    alpha_T = k / (rho * cp)
    nu = mu / rho

    # diffusivités espèces
    D_default = 2e-9
    D = np.full(nspecies, D_default, dtype=float)
    dspec = data.get("matDiffusivitySpecs", {}).get("LiPb", "")
    if dspec:
        parts = [p.strip() for p in str(dspec).split(",")]
        for s in range(min(nspecies, len(parts))):
            try:
                D[s] = float(parts[s])
            except Exception:
                pass

    reactions = parse_reactions(data)
    Bz = get_imposed_B_from_json(data)

    # pseudo-temps (CFL adaptatif)
    # dt_base = 0.2 * min(dx, dy)
    dt_base = 2e-4
    dt = dt_base

    relax_u = 0.7
    relax_p = 0.2
    relax_T = 1.0
    relax_C = 0.9

    use_convection_ns = False

    print(f"[RUN] dx={dx:g}m dy={dy:g}m dt0={dt:g} tol={tol} iters={itmax}")
    print(f"[MAT] rho={rho:g} mu={mu:g} nu={nu:g} k={k:g} cp={cp:g} alpha_T={alpha_T:g} sigma={sigma_e:g} Bz={Bz:g}")
    print(f"[CHEM] reactions={len(reactions)} species={SPECIES}")

    Ny, Nx = mask.shape

    for it in range(1, itmax + 1):
        ux_old = fields.ux.copy()
        uy_old = fields.uy.copy()
        p_old = fields.p.copy()
        T_old = fields.T.copy()
        C_old = fields.C.copy()

        # dt adaptatif selon vitesse
        speed = np.zeros_like(fields.ux)
        speed[mask] = np.sqrt(fields.ux[mask] ** 2 + fields.uy[mask] ** 2)
        umax = float(np.nanmax(speed[mask])) if np.any(mask) else 0.0
        if umax > 1e-12:
            dt_adv = 0.5 * min(dx, dy) / umax
            dt = min(dt_base, dt_adv)
        else:
            dt = dt_base

        (
            dir_u, dir_ux_val, dir_uy_val,
            dir_p, dir_p_val,
            dir_T, dir_T_val,
            dir_C, dir_C_val,
            flux_ux_faces, flux_uy_faces,
            flux_T_faces,
            flux_C_faces
        ) = apply_bcs_build_masks_and_fluxes(data, fields, bnodes, mask)


        # A) NS Stokes + MHD
        fx_mhd, fy_mhd = mhd_lorentz_force(fields.ux, fields.uy, Bz, sigma_e, mask)

        if use_convection_ns:
            conv_ux = upwind_convection(fields.ux, fields.ux, fields.uy, mask, dx, dy)
            conv_uy = upwind_convection(fields.uy, fields.ux, fields.uy, mask, dx, dy)
        else:
            conv_ux = 0.0
            conv_uy = 0.0

        gp_x, gp_y = grad_scalar(fields.p, mask, dx, dy)
        rhs_ux = (-conv_ux) + (-gp_x / rho) + (fx_mhd / rho)
        rhs_uy = (-conv_uy) + (-gp_y / rho) + (fy_mhd / rho)
        
        # --- ux ---
        fields.ux = diffusion_implicit_step_with_flux(
            fields.ux, rhs_ux, mask, dx, dy, alpha=nu, dt=dt,
            dirichlet_mask=dir_u, dirichlet_value=dir_ux_val,
            neumann_flux_faces=flux_ux_faces, K_for_flux=1.0,
            n_iter=250, omega=1.25,
            tol_inner=1e-6, check_every=10
        )
        
        # --- uy ---
        fields.uy = diffusion_implicit_step_with_flux(
            fields.uy, rhs_uy, mask, dx, dy, alpha=nu, dt=dt,
            dirichlet_mask=dir_u, dirichlet_value=dir_uy_val,
            neumann_flux_faces=flux_uy_faces, K_for_flux=1.0,
            n_iter=250, omega=1.25,
            tol_inner=1e-6, check_every=10
        )



        fields.ux = ux_old + relax_u * (fields.ux - ux_old)
        fields.uy = uy_old + relax_u * (fields.uy - uy_old)

        sanitize_inplace(fields.ux, mask, clip_abs=1e6)
        sanitize_inplace(fields.uy, mask, clip_abs=1e6)

        divu = compute_divergence(fields.ux, fields.uy, mask, dx, dy)
        rhs_p = (rho / max(dt, 1e-30)) * divu
        fields.p = poisson_sor(
            fields.p, rhs_p, mask, dx, dy,
            fixed_mask=dir_p,
            fixed_value_array=dir_p_val,
            n_iter=600, omega=1.6,
            tol_inner=1e-6, check_every=10
        )



        gp_x, gp_y = grad_scalar(fields.p, mask, dx, dy)
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if not mask[j, i] or dir_u[j, i]:
                    continue
                fields.ux[j, i] -= (dt / rho) * gp_x[j, i]
                fields.uy[j, i] -= (dt / rho) * gp_y[j, i]

        fields.p = p_old + relax_p * (fields.p - p_old)

        sanitize_inplace(fields.p, mask, clip_abs=1e12)

        # B) Réactions
        if reactions:
            S_C, Q_vol = compute_reaction_sources(fields.C, fields.T, reactions)
        else:
            S_C = np.zeros_like(fields.C)
            Q_vol = np.zeros_like(fields.T)

        # C) Energie (T) + flux
        conv_T = upwind_convection(fields.T, fields.ux, fields.uy, mask, dx, dy)
        rhs_T = conv_T + (Q_vol / (rho * cp))

        fields.T = diffusion_implicit_step_with_flux(
            fields.T, rhs_T, mask, dx, dy, alpha=alpha_T, dt=dt,
            dirichlet_mask=dir_T, dirichlet_value=dir_T_val,
            neumann_flux_faces=flux_T_faces, K_for_flux=k,
            n_iter=600, omega=1.25
        )
        fields.T = T_old + relax_T * (fields.T - T_old)
        sanitize_inplace(fields.T, mask, clip_abs=1e7)

        # D) Espèces + flux + réactions
        for s in range(nspecies):
            conv_C = upwind_convection(fields.C[s], fields.ux, fields.uy, mask, dx, dy)
            rhs_C = conv_C + S_C[s]
            fields.C[s] = diffusion_implicit_step_with_flux(
                fields.C[s], rhs_C, mask, dx, dy, alpha=D[s], dt=dt,
                dirichlet_mask=dir_C[s], dirichlet_value=dir_C_val[s],
                neumann_flux_faces=flux_C_faces[s], K_for_flux=D[s],
                n_iter=350, omega=1.25,
                tol_inner=1e-6, check_every=10
            )

            fields.C[s] = C_old[s] + relax_C * (fields.C[s] - C_old[s])
            sanitize_inplace(fields.C[s], mask, clip_abs=1e6)

        # E) Convergence
        du = max(np.nanmax(np.abs(fields.ux - ux_old)), np.nanmax(np.abs(fields.uy - uy_old)))
        dp = np.nanmax(np.abs(fields.p - p_old))
        dT = np.nanmax(np.abs(fields.T - T_old))
        dC = np.nanmax(np.abs(fields.C - C_old))
        div_norm = np.nanmax(np.abs(compute_divergence(fields.ux, fields.uy, mask, dx, dy)))

        if it % 20 == 0 or it == 1:
            print(f"[it={it:4d}] dt={dt:.3e} du={du:.3e} dp={dp:.3e} dT={dT:.3e} dC={dC:.3e} | max|div u|={div_norm:.3e}")

        if max(du, dp, dT, dC) < tol:
            print(f"[OK] Convergence atteinte it={it} (max delta={max(du,dp,dT,dC):.3e})")
            break

    # =======================================================
    # Post-traitement
    xmin, xmax, ymin, ymax = extent

    plt.figure(figsize=(7, 6))
    Tplot = np.where(mask, fields.T, np.nan)
    plt.imshow(Tplot, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="equal", interpolation="nearest")
    plt.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2)
    plt.title("Température T (flux + sources)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(7, 6))
    step = 1
    Uq = np.where(mask, fields.ux, np.nan)
    Vq = np.where(mask, fields.uy, np.nan)
    plt.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2)
    plt.quiver(X[::step, ::step], Y[::step, ::step], Uq[::step, ::step], Vq[::step, ::step])
    plt.title("Vitesse u (projection + MHD simplifiée)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    plt.figure(figsize=(7, 6))
    nf = np.where(mask, fields.C[S_INDEX["nfast"]], np.nan)
    plt.imshow(nf, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="equal", interpolation="nearest")
    plt.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2)
    plt.title("Espèce nfast (flux + réactions)")
    plt.colorbar()
    plt.tight_layout()

    plt.show()

    return fields


# ============================================================
# Main
# ============================================================

def main():
    data = load_json("projetTest.json")

    domain_name = "Domaine_1"
    mesh_kind, dx, dy = get_mesh_steps(data, domain_name=domain_name)
    if mesh_kind != "TRI_REG":
        print(f"[INFO] kind={mesh_kind}. Le script attend TRI_REG.")

    poly_pts = get_domain_polygon(data, domain_name=domain_name)

    X, Y, mask, extent, path = build_grid_and_mask(poly_pts, dx=dx, dy=dy, padding=0.0)
    tri = build_triangulation_from_grid(X, Y, path)

    plot_mesh(poly_pts, X, Y, mask, tri, extent)
    print("Grille:", X.shape, "- inside:", int(mask.sum()))

    run_coupled_solver(data, X, Y, mask, poly_pts, tri, extent)


if __name__ == "__main__":
    main()
