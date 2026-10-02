"""Pure geometry for the holographic globe HUD (hud/globe_hud.py).

No tkinter, no I/O, no state: floats and tuples in, floats and tuples out, so
the whole projection / culling / arc / easing pipeline is unit-testable on a
display-less CI runner (tests/test_globe_geometry.py).

Conventions
  * lat / lon in DEGREES, lat north-positive, lon east-positive.
  * A point on the unit sphere is (x, y, z): x toward (0N, 0E), y toward
    (0N, 90E), z toward the north pole.
  * A VIEW is the (lat, lon) at the centre of the visible disc. The globe
    spins by changing the view's lon; the view's lat tilts it.
  * Projection is ORTHOGRAPHIC: a rotated point (e, n, depth) lands at (e, n)
    in the unit disc (e right, n up); depth > 0 is the hemisphere facing the
    viewer. Screen pixels are cx + R*e, cy - R*n (tkinter's y runs down).
"""
from __future__ import annotations

import math

# Graticule spacing: a meridian and a parallel every 15 degrees.
GRATICULE_STEP_DEG = 15


# ─── sphere <-> lat/lon ───────────────────────────────────────────────────

def latlon_to_xyz(lat: float, lon: float, radius: float = 1.0) -> tuple:
    """(lat, lon) in degrees -> a point on a sphere of `radius`."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    c = math.cos(phi) * radius
    return (c * math.cos(lam), c * math.sin(lam), math.sin(phi) * radius)


def xyz_to_latlon(x: float, y: float, z: float) -> tuple:
    """Any non-zero (x, y, z) -> its (lat, lon) in degrees, lon in [-180, 180]."""
    r = math.sqrt(x * x + y * y + z * z) or 1.0
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def wrap_lon(lon: float) -> float:
    """Normalise a longitude into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def shortest_lon_delta(start: float, end: float) -> float:
    """Signed degrees to turn from `start` to `end` the short way round,
    in [-180, 180)."""
    return wrap_lon(end - start)


# ─── the view rotation ────────────────────────────────────────────────────

def view_basis(view_lat: float, view_lon: float) -> tuple:
    """Precompute the sines/cosines a view needs: (sin lat0, cos lat0,
    sin lon0, cos lon0). Compute once per frame, reuse for every point."""
    p = math.radians(view_lat)
    l = math.radians(view_lon)
    return (math.sin(p), math.cos(p), math.sin(l), math.cos(l))


def rotate(point: tuple, basis: tuple) -> tuple:
    """Rotate a sphere point into view space: (e, n, depth). (e, n) is the
    orthographic position in the unit disc; depth > 0 faces the viewer."""
    x, y, z = point
    sp, cp, sl, cl = basis
    d = x * cl + y * sl          # component toward the view's meridian
    e = y * cl - x * sl          # east, screen right
    return (e, cp * z - sp * d, sp * z + cp * d)


def unrotate(e: float, n: float, depth: float, basis: tuple) -> tuple:
    """Inverse of rotate(): view space back to sphere (x, y, z)."""
    sp, cp, sl, cl = basis
    z = cp * n + sp * depth
    d = cp * depth - sp * n
    return (d * cl - e * sl, d * sl + e * cl, z)


def project(lat: float, lon: float, view_lat: float, view_lon: float) -> tuple:
    """Orthographic projection of (lat, lon) for a view centred on
    (view_lat, view_lon): (e, n, visible). (e, n) lies in the unit disc;
    `visible` is False on the far hemisphere (back-face culled)."""
    e, n, depth = rotate(latlon_to_xyz(lat, lon), view_basis(view_lat, view_lon))
    return e, n, depth >= 0.0


def unproject(e: float, n: float, view_lat: float, view_lon: float):
    """Inverse of project() for a near-side point: the (lat, lon) under disc
    position (e, n), or None outside the disc."""
    rho2 = e * e + n * n
    if rho2 > 1.0 + 1e-9:
        return None
    depth = math.sqrt(max(0.0, 1.0 - rho2))
    return xyz_to_latlon(*unrotate(e, n, depth, view_basis(view_lat, view_lon)))


def _visibility(e: float, n: float, depth: float) -> float:
    """>= 0 when the point is visible. Near-side points always are. A far-side
    point is hidden behind the sphere unless it sits outside the disc's
    silhouette (only possible for an arc lifted above the surface)."""
    if depth >= 0.0:
        return depth
    r = e * e + n * n - 1.0
    return r if r > depth else depth


def is_visible(point: tuple, basis: tuple) -> bool:
    """Back-face test for one sphere point (lifted points included)."""
    return _visibility(*rotate(point, basis)) >= 0.0


def project_runs(polylines, basis: tuple, cx: float, cy: float,
                 radius: float) -> list:
    """Project polylines of sphere points to screen space, back-face culled.

    Each polyline is split into RUNS of consecutive visible points; every run
    comes back as a flat [x0, y0, x1, y1, ...] list of at least two points,
    ready for a canvas line. Where a run meets the horizon, the crossing is
    interpolated so the line ends on the limb instead of one segment short.
    The hot loop of the HUD: kept to plain arithmetic, one pass per point."""
    sp, cp, sl, cl = basis
    out = []
    for line in polylines:
        run = None
        pe = pn = pg = 0.0
        first = True
        for x, y, z in line:
            d = x * cl + y * sl
            e = y * cl - x * sl
            n = cp * z - sp * d
            g = sp * z + cp * d
            if g < 0.0:
                r = e * e + n * n - 1.0
                if r > g:
                    g = r
            if g >= 0.0:
                if run is None:
                    run = []
                    if not first and pg < 0.0:
                        t = pg / (pg - g)
                        run += (cx + radius * (pe + (e - pe) * t),
                                cy - radius * (pn + (n - pn) * t))
                run += (cx + radius * e, cy - radius * n)
            elif run is not None:
                t = pg / (pg - g)
                run += (cx + radius * (pe + (e - pe) * t),
                        cy - radius * (pn + (n - pn) * t))
                if len(run) >= 4:
                    out.append(run)
                run = None
            pe, pn, pg = e, n, g
            first = False
        if run is not None and len(run) >= 4:
            out.append(run)
    return out


def to_screen(point: tuple, basis: tuple, cx: float, cy: float,
              radius: float):
    """One sphere point -> (sx, sy) on screen, or None when back-face culled."""
    e, n, depth = rotate(point, basis)
    if _visibility(e, n, depth) < 0.0:
        return None
    return cx + radius * e, cy - radius * n


# ─── static line sets ─────────────────────────────────────────────────────

def coastline_polylines(lines) -> list:
    """Decode the bundled coastline ([lon, lat, lon, lat, ...] per line, see
    hud/data/globe_coastline.json) into polylines of sphere points."""
    out = []
    for flat in lines:
        pts = [latlon_to_xyz(flat[i + 1], flat[i])
               for i in range(0, len(flat) - 1, 2)]
        if len(pts) >= 2:
            out.append(pts)
    return out


def meridians(step_deg: float = GRATICULE_STEP_DEG,
              sample_deg: float = 5.0) -> list:
    """Lines of longitude every `step_deg`, pole to pole."""
    lats = _samples(-90.0, 90.0, sample_deg)
    return [[latlon_to_xyz(lat, lon) for lat in lats]
            for lon in _samples(-180.0, 180.0 - step_deg, step_deg)]


def parallels(step_deg: float = GRATICULE_STEP_DEG,
              sample_deg: float = 5.0) -> list:
    """Closed lines of latitude every `step_deg` (equator included, poles
    skipped - they are points)."""
    lons = _samples(-180.0, 180.0, sample_deg)
    return [[latlon_to_xyz(lat, lon) for lon in lons]
            for lat in _samples(-90.0 + step_deg, 90.0 - step_deg, step_deg)]


def _samples(start: float, stop: float, step: float) -> list:
    n = int(round((stop - start) / step))
    return [start + i * step for i in range(n + 1)]


# ─── great-circle arcs ────────────────────────────────────────────────────

def angular_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Central angle between two points, in degrees."""
    a = latlon_to_xyz(lat1, lon1)
    b = latlon_to_xyz(lat2, lon2)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def great_circle_arc(lat1: float, lon1: float, lat2: float, lon2: float,
                     step_deg: float = 3.0, lift: float = 0.0) -> list:
    """Sphere points along the great circle from point 1 to point 2.

    Spherical linear interpolation, one sample about every `step_deg`. The
    first and last points are exactly the endpoints. `lift` raises the middle
    of the arc above the surface (radius 1 + lift at the midpoint, 1 at both
    ends) for the holographic flight-path look. Antipodal endpoints have no
    unique great circle; one through the pole-ward perpendicular is used."""
    a = latlon_to_xyz(lat1, lon1)
    b = latlon_to_xyz(lat2, lon2)
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    omega = math.acos(dot)
    if omega < 1e-9:
        return [a, b]
    steps = max(2, int(math.ceil(math.degrees(omega) / max(0.1, step_deg))))
    sin_o = math.sin(omega)
    if sin_o < 1e-6:
        # Antipodal: walk a half great circle via any unit vector u
        # perpendicular to a: p(t) = a cos(pi t) + u sin(pi t).
        ref = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
        u = _normalise(_cross(_cross(a, ref), a))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            c, s = math.cos(math.pi * t), math.sin(math.pi * t)
            pts.append(tuple(a[k] * c + u[k] * s for k in range(3)))
    else:
        pts = []
        for i in range(steps + 1):
            t = i / steps
            wa = math.sin((1.0 - t) * omega) / sin_o
            wb = math.sin(t * omega) / sin_o
            pts.append((a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb,
                        a[2] * wa + b[2] * wb))
    pts[0], pts[-1] = a, b
    if lift > 0.0:
        for i in range(1, steps):
            k = 1.0 + lift * math.sin(math.pi * i / steps)
            p = pts[i]
            pts[i] = (p[0] * k, p[1] * k, p[2] * k)
    return pts


def _cross(a: tuple, b: tuple) -> tuple:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _normalise(v: tuple) -> tuple:
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


# ─── rotation easing ──────────────────────────────────────────────────────

def ease_in_out_cubic(t: float) -> float:
    """Smooth 0 -> 1 with zero velocity at both ends; t is clamped to
    [0, 1], so the result never leaves [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    u = -2.0 * t + 2.0
    return 1.0 - u * u * u / 2.0


def tween_view(start: tuple, end: tuple, t: float) -> tuple:
    """The (lat, lon) view a fraction `t` of the way from `start` to `end`,
    eased, turning the short way round in longitude."""
    k = ease_in_out_cubic(t)
    lat = start[0] + (end[0] - start[0]) * k
    lon = start[1] + shortest_lon_delta(start[1], end[1]) * k
    return lat, wrap_lon(lon)


def approach(current: float, target: float, max_step: float) -> float:
    """Move `current` toward `target` by at most `max_step` (never past it)."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


def degrees_per_pixel(radius_px: float) -> float:
    """How far the globe turns for a point at the disc centre to move one
    pixel - the HUD skips redrawing the wireframe below this."""
    return math.degrees(1.0 / max(1.0, radius_px))
