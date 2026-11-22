"""Metric service-core geometry for the ML environment, not a regulatory model.

One source defines components, raster reservations, API data and SVG drawings.
Height bands and lift counts are experimental priors, not traffic/egress design.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from html import escape
import math

CORE_LAYOUT_VERSION = "compact-core.v5"


@dataclass(frozen=True)
class CorePlanParameters:
    floor_height: float = 3.0
    risers_per_flight: int = 9
    tread: float = 0.28
    flight_width: float = 1.20
    landing_depth: float = 1.20
    stair_gap: float = 0.10
    wall: float = 0.15
    shaft_width: float = 2.10  # outside, including walls
    shaft_depth: float = 2.20
    car_width: float = 1.10
    car_depth: float = 1.40
    door_width: float = 0.80
    stair_separator: float = 0.15
    stair_slab: float = 0.15
    lobby_clear_width: float = 1.20

    def __post_init__(self):
        values = (
            self.floor_height,
            self.tread,
            self.flight_width,
            self.landing_depth,
            self.wall,
            self.shaft_width,
            self.shaft_depth,
            self.car_width,
            self.car_depth,
            self.door_width,
            self.stair_separator,
            self.stair_slab,
            self.lobby_clear_width,
        )
        if (
            any(not math.isfinite(v) or v <= 0 for v in values)
            or self.risers_per_flight < 2
            or self.stair_gap < 0
        ):
            raise ValueError("Core geometry must be finite and positive")
        if self.landing_depth < self.flight_width:
            raise ValueError("Landing depth must accommodate a flight width")
        if (
            self.shaft_width - 2 * self.wall < self.car_width
            or self.shaft_depth - 2 * self.wall < self.car_depth
        ):
            raise ValueError("Lift car must fit inside the shaft walls")
        if self.door_width > min(self.car_width, self.flight_width):
            raise ValueError("Door must fit the car and stair")


@dataclass(frozen=True)
class HeightBand:
    tier: str
    minimum: int
    maximum: int
    stairs: int
    lifts: int
    stair_layout: str = "dogleg"


SCISSOR_HEIGHT_BAND = HeightBand("T4", 19, 30, 2, 4, "scissor")

DEFAULT_HEIGHT_BANDS = (
    HeightBand("T1", 1, 5, 1, 0),
    HeightBand("T2", 6, 10, 1, 1),
    HeightBand("T3", 11, 18, 1, 2),
    SCISSOR_HEIGHT_BAND,
    replace(SCISSOR_HEIGHT_BAND, tier="T5", minimum=31, maximum=60),
)


def make_plan(band: HeightBand, p: CorePlanParameters | None = None) -> dict:
    p = p or CorePlanParameters()
    if (
        band.stairs not in (1, 2)
        or band.lifts < 0
        or band.minimum < 1
        or band.maximum < band.minimum
    ):
        raise ValueError("Unsupported height band")
    if band.stair_layout == "scissor":
        return make_scissor_plan(band, p)
    if band.stair_layout != "dogleg":
        raise ValueError("Unknown stair layout")
    run = (p.risers_per_flight - 1) * p.tread
    sw = 2 * p.wall + 2 * p.flight_width + p.stair_gap
    length = 2 * p.wall + 2 * p.landing_depth + run
    lift_bank = band.lifts * p.shaft_width
    width = band.stairs * sw + lift_bank
    zones = []
    parts = []
    doors = []

    def rect(kind, x, y, w, h, label, **extra):
        return dict(
            type=kind,
            x=round(x, 6),
            y=round(y, 6),
            w=round(w, 6),
            h=round(h, 6),
            label=label,
            **extra,
        )

    stair_xs = [0.0] if band.stairs == 1 else [0.0, sw + lift_bank]
    for i, x in enumerate(stair_xs):
        zones.append(rect("stair", x, 0, sw, length, f"Stair {i + 1}"))
        parts.append(
            rect(
                "landing",
                x + p.wall,
                p.wall,
                sw - 2 * p.wall,
                p.landing_depth,
                "Floor landing",
                stair=i,
            )
        )
        parts.append(
            rect(
                "landing",
                x + p.wall,
                p.wall + p.landing_depth + run,
                sw - 2 * p.wall,
                p.landing_depth,
                "Half landing",
                stair=i,
            )
        )
        for flight in (0, 1):
            parts.append(
                rect(
                    "flight",
                    x + p.wall + flight * (p.flight_width + p.stair_gap),
                    p.wall + p.landing_depth,
                    p.flight_width,
                    run,
                    f"Flight {flight + 1}",
                    stair=i,
                    risers=p.risers_per_flight,
                    treads=p.risers_per_flight - 1,
                    direction=1 if flight == 0 else -1,
                )
            )
        # Side opening joins the floor landing directly to the adjacent lobby.
        if band.lifts:
            doors.append(
                dict(
                    type="stair_door",
                    x=round(x + sw if i == 0 else x, 6),
                    y=round(p.wall + 0.15, 6),
                    width=0.9,
                    orientation="vertical",
                )
            )
        else:
            doors.append(
                dict(
                    type="entrance",
                    x=round(x + 0.95, 6),
                    y=0.0,
                    width=0.9,
                    orientation="horizontal",
                )
            )
    if band.lifts:
        lobby_depth = length - p.shaft_depth
        zones.append(rect("lobby", sw, 0, lift_bank, lobby_depth, "Shared lobby"))
        doors.append(
            dict(
                type="entrance",
                x=round(sw + (lift_bank - 0.9) / 2, 6),
                y=0.0,
                width=0.9,
                orientation="horizontal",
            )
        )
        for i in range(band.lifts):
            x = sw + i * p.shaft_width
            y = lobby_depth
            zones.append(
                rect("lift", x, y, p.shaft_width, p.shaft_depth, f"Lift {i + 1}")
            )
            parts.append(
                rect(
                    "car",
                    x + (p.shaft_width - p.car_width) / 2,
                    y + p.wall + 0.30,
                    p.car_width,
                    p.car_depth,
                    f"Car {i + 1}",
                )
            )
            doors.append(
                dict(
                    type="lift_door",
                    x=round(x + (p.shaft_width - p.door_width) / 2, 6),
                    y=round(y, 6),
                    width=p.door_width,
                    orientation="horizontal",
                )
            )
    width = round(width, 6)
    length = round(length, 6)
    return dict(
        version=CORE_LAYOUT_VERSION,
        tier=band.tier,
        minHeight=band.minimum,
        maxHeight=band.maximum,
        stairCount=band.stairs,
        stairLayout="dogleg",
        stairEnclosureCount=band.stairs,
        liftCount=band.lifts,
        metricWidth=width,
        metricLength=length,
        metricArea=round(width * length, 4),
        width=math.ceil(width - 1e-9),
        length=math.ceil(length - 1e-9),
        floorHeight=p.floor_height,
        riserCount=2 * p.risers_per_flight,
        riserHeight=p.floor_height / (2 * p.risers_per_flight),
        tread=p.tread,
        clearFlightWidth=p.flight_width,
        landingDepth=p.landing_depth,
        wallThickness=p.wall,
        shaftClearWidth=p.shaft_width - 2 * p.wall,
        shaftClearDepth=p.shaft_depth - 2 * p.wall,
        carWidth=p.car_width,
        carDepth=p.car_depth,
        zones=zones,
        components=parts,
        doors=doors,
        assumptions="Research layout prior; lift counts depend on this configured height schedule. Not a code, traffic or evacuation assessment.",
    )


def make_scissor_plan(band: HeightBand, p: CorePlanParameters) -> dict:
    """Two interlocking routes; straight half-flights split by a mid-landing.

    Lane 0 always rises towards the far end; lane 1 towards the near end.
    Routes change lanes at the full-width storey landings, so each route
    exchanges ends on successive storeys. Flights stacked in the same lane
    have a full storey's clearance, not the impossible half-storey overlap
    produced by superimposing two compact doglegs.
    """
    if band.stairs != 2 or band.lifts < 1:
        raise ValueError("Scissor prior requires two routes and a lift bank")
    if p.floor_height - p.stair_slab < 2.0:
        raise ValueError("Scissor section has less than the assumed 2 m headroom")
    run = (p.risers_per_flight - 1) * p.tread
    sw = 2 * p.wall + 2 * p.flight_width + p.stair_separator
    stair_length = 2 * p.wall + 3 * p.landing_depth + 2 * run
    length = max(stair_length, band.lifts * p.shaft_width)
    # Distribute any extra bank length across the end landings; flights and
    # the intermediate landings retain their original run/depth.
    end_landing = p.landing_depth + (length - stair_length) / 2
    lobby_width = p.lobby_clear_width + 2 * p.wall
    width = sw + lobby_width + p.shaft_depth

    def rect(kind, x, y, w, h, label, **extra):
        return dict(
            type=kind,
            x=round(x, 6),
            y=round(y, 6),
            w=round(w, 6),
            h=round(h, 6),
            label=label,
            **extra,
        )

    zones = [
        rect("stair", 0, 0, sw, length, "Scissor enclosure: routes A/B"),
        rect("lobby", sw, 0, lobby_width, length, "Lift lobby"),
    ]
    if length - band.lifts * p.shaft_width > 1e-8:
        zones.append(
            rect(
                "service",
                sw + lobby_width,
                band.lifts * p.shaft_width,
                p.shaft_depth,
                length - band.lifts * p.shaft_width,
                "Service reserve",
            )
        )
    parts = []
    doors = []
    routes = []
    middle = p.wall + end_landing + run
    far = length - p.wall - end_landing
    for lane, route in enumerate(("A", "B")):
        x = p.wall + lane * (p.flight_width + p.stair_separator)
        start_y = p.wall if lane == 0 else far
        parts.append(
            rect(
                "landing",
                p.wall,
                start_y,
                sw - 2 * p.wall,
                end_landing,
                f"{route}: floor landing",
                route=route,
                elevation=0.0,
                level="floor",
            )
        )
        parts.append(
            rect(
                "landing",
                x,
                middle,
                p.flight_width,
                p.landing_depth,
                f"{route}: +{p.floor_height / 2:g} m",
                route=route,
                elevation=p.floor_height / 2,
                level="intermediate",
            )
        )
        for j, y in enumerate((p.wall + end_landing, middle + p.landing_depth)):
            low = j if lane == 0 else 1 - j
            parts.append(
                rect(
                    "flight",
                    x,
                    y,
                    p.flight_width,
                    run,
                    f"{route}: flight {low + 1}",
                    route=route,
                    risers=p.risers_per_flight,
                    treads=p.risers_per_flight - 1,
                    direction=1 if lane == 0 else -1,
                    zStart=low * p.floor_height / 2,
                    zEnd=(low + 1) * p.floor_height / 2,
                )
            )
        nodes = [
            dict(lane=lane, end="near" if lane == 0 else "far", z=0.0),
            dict(lane=lane, end="middle", z=p.floor_height / 2),
            dict(lane=1 - lane, end="far" if lane == 0 else "near", z=p.floor_height),
        ]
        routes.append(
            dict(
                id=route,
                laneOnEvenStorey=lane,
                laneOnOddStorey=1 - lane,
                nodes=nodes,
                risersPerStorey=2 * p.risers_per_flight,
            )
        )
        doors.append(
            dict(
                type="stair_door",
                route=route,
                x=round(sw, 6),
                y=round(start_y + 0.15, 6),
                width=0.9,
                orientation="vertical",
            )
        )
    for i in range(band.lifts):
        x = sw + lobby_width
        y = i * p.shaft_width
        zones.append(rect("lift", x, y, p.shaft_depth, p.shaft_width, f"Lift {i + 1}"))
        parts.append(
            rect(
                "car",
                x + p.wall + 0.30,
                y + (p.shaft_width - p.car_width) / 2,
                p.car_depth,
                p.car_width,
                f"Car {i + 1}",
            )
        )
        doors.append(
            dict(
                type="lift_door",
                x=round(x, 6),
                y=round(y + (p.shaft_width - p.door_width) / 2, 6),
                width=p.door_width,
                orientation="vertical",
            )
        )
    doors.append(
        dict(
            type="entrance",
            x=round(sw + (lobby_width - 0.9) / 2, 6),
            y=0.0,
            width=0.9,
            orientation="horizontal",
        )
    )
    width = round(width, 6)
    length = round(length, 6)
    return dict(
        version=CORE_LAYOUT_VERSION,
        tier=band.tier,
        minHeight=band.minimum,
        maxHeight=band.maximum,
        stairCount=2,
        stairLayout="scissor",
        stairEnclosureCount=1,
        liftCount=band.lifts,
        metricWidth=width,
        metricLength=length,
        metricArea=round(width * length, 4),
        width=math.ceil(width - 1e-9),
        length=math.ceil(length - 1e-9),
        floorHeight=p.floor_height,
        riserCount=2 * p.risers_per_flight,
        riserHeight=p.floor_height / (2 * p.risers_per_flight),
        tread=p.tread,
        clearFlightWidth=p.flight_width,
        landingDepth=p.landing_depth,
        wallThickness=p.wall,
        separatorThickness=p.stair_separator,
        stairSlabThickness=p.stair_slab,
        endLandingDepth=end_landing,
        clearStackedHeadroom=p.floor_height - p.stair_slab,
        clearLobbyWidth=p.lobby_clear_width,
        shaftClearWidth=p.shaft_width - 2 * p.wall,
        shaftClearDepth=p.shaft_depth - 2 * p.wall,
        carWidth=p.car_width,
        carDepth=p.car_depth,
        zones=zones,
        components=parts,
        doors=doors,
        routes=routes,
        assumptions="Research scissor prior: two routes in one enclosure, separated lanes and intermediate landings. Not an egress independence or compliance assessment.",
    )


def plan_library():
    return [make_plan(b) for b in DEFAULT_HEIGHT_BANDS]


def render_plan_svg(plan: dict) -> str:
    """Scaled vector plan with separate flights, landings, car/shaft and openings."""
    scale = 50.0
    ox = 70.0
    oy = 85.0
    w = plan["metricWidth"]
    h = plan["metricLength"]
    vw = max(390.0, ox + w * scale + 80)
    vh = oy + h * scale + 130
    if plan.get("stairLayout") == "scissor":
        vh += 155
    stair_label = (
        "2 interlocking routes"
        if plan.get("stairLayout") == "scissor"
        else f"{plan['stairCount']} stair(s)"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:g} {vh:g}" role="img" aria-label="{plan["tier"]} core plan">',
        '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#1645ff"/></marker></defs>',
        f'<rect width="{vw:g}" height="{vh:g}" fill="white"/>',
        f'<g font-family="Arial,sans-serif" fill="#182238"><text x="20" y="25" font-size="16" font-weight="bold">{plan["tier"]} · {plan["minHeight"]}–{plan["maxHeight"]} floors</text>',
        f'<text x="20" y="46" font-size="11">{stair_label} · {plan["liftCount"]} lift(s) · {plan["metricArea"]:g} m² metric / {plan["width"] * plan["length"]} m² grid reservation</text>',
    ]

    def line(x1, y1, x2, y2, stroke="#344158", width=1, extra=""):
        out.append(
            f'<line x1="{ox + x1 * scale:g}" y1="{oy + y1 * scale:g}" x2="{ox + x2 * scale:g}" y2="{oy + y2 * scale:g}" stroke="{stroke}" stroke-width="{width}" {extra}/>'
        )

    def rect(c, fill, stroke="#344158", width=1):
        out.append(
            f'<rect x="{ox + c["x"] * scale:g}" y="{oy + c["y"] * scale:g}" width="{c["w"] * scale:g}" height="{c["h"] * scale:g}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    # Dashed outline shows the conservative one-metre solver reservation.
    rect(dict(x=0, y=0, w=plan["width"], h=plan["length"]), "#fafbff", "#b8c7ef")
    for z in plan["zones"]:
        rect(
            z,
            {
                "stair": "#e2e8f0",
                "lobby": "#fff5d8",
                "lift": "#c7d2fe",
                "service": "#f1f5f9",
            }[z["type"]],
            width=2,
        )
        if z["type"] in ("stair", "lift"):
            t = plan["wallThickness"]
            rect(
                dict(x=z["x"] + t, y=z["y"] + t, w=z["w"] - 2 * t, h=z["h"] - 2 * t),
                "white",
            )
        elif z["type"] == "lobby":
            out.append(
                f'<text x="{ox + (z["x"] + z["w"] * 0.5) * scale:g}" y="{oy + (z["y"] + z["h"] * 0.55) * scale:g}" text-anchor="middle" font-size="11">Lobby</text>'
            )
    if plan.get("stairLayout") == "scissor":
        stair = plan["zones"][0]
        x = (
            plan["wallThickness"]
            + plan["clearFlightWidth"]
            + plan["separatorThickness"] / 2
        )
        line(
            x,
            plan["wallThickness"] + plan["landingDepth"],
            x,
            stair["h"] - plan["wallThickness"] - plan["landingDepth"],
            width=plan["separatorThickness"] * scale,
        )
    for c in plan["components"]:
        if c["type"] == "landing":
            rect(c, "#f0f6ff", "#94a3b8")
            out.append(
                f'<text x="{ox + (c["x"] + c["w"] * 0.5) * scale:g}" y="{oy + (c["y"] + c["h"] * 0.55) * scale:g}" text-anchor="middle" font-size="8">{escape(c["label"])}</text>'
            )
        elif c["type"] == "flight":
            rect(c, "white", "#94a3b8")
            for j in range(c["treads"] + 1):
                y = c["y"] + c["h"] * j / c["treads"]
                line(c["x"], y, c["x"] + c["w"], y, stroke="#64748b", width=0.8)
            mid = c["x"] + c["w"] / 2
            y0 = c["y"] + 0.2
            y1 = c["y"] + c["h"] - 0.2
            if c["direction"] < 0:
                y0, y1 = y1, y0
            line(
                mid,
                y0,
                mid,
                y1,
                stroke="#1645ff",
                width=1.8,
                extra='marker-end="url(#arrow)"',
            )
            out.append(
                f'<text x="{ox + mid * scale:g}" y="{oy + (c["y"] + c["h"] * 0.55) * scale:g}" font-size="8" fill="#1645ff">UP</text>'
            )
        elif c["type"] == "car":
            rect(c, "#eef2ff", "#4338ca", 1.3)
    for door in plan["doors"]:
        x = door["x"]
        y = door["y"]
        d = door["width"]
        if door["orientation"] == "horizontal":
            line(
                x,
                y,
                x + d,
                y,
                stroke="white",
                width=plan["wallThickness"] * scale * 2 + 2,
            )
            line(x, y, x + d, y, stroke="#16a34a", width=2)
            if door["type"] == "entrance":
                line(x, y, x, y - 0.7, stroke="#344158", width=1)
        else:
            line(
                x,
                y,
                x,
                y + d,
                stroke="white",
                width=plan["wallThickness"] * scale * 2 + 2,
            )
            line(x, y, x, y + d, stroke="#16a34a", width=2)
    # Metric dimensions rather than inferring scale from a rectangle label.
    line(0, -0.3, w, -0.3)
    line(-0.3, 0, -0.3, h)
    out.append(
        f'<text x="{ox + w * scale / 2:g}" y="{oy - 23:g}" text-anchor="middle" font-size="11">{w:.2f} m</text>'
    )
    out.append(
        f'<text transform="translate({ox - 26:g},{oy + h * scale / 2:g}) rotate(-90)" text-anchor="middle" font-size="11">{h:.2f} m</text>'
    )
    if plan.get("stairLayout") == "scissor":
        sy = oy + h * scale + 40
        sh = 55
        left = ox
        span = w * scale
        out.append(
            f'<text x="20" y="{sy - 15:g}" font-size="10">Developed scissor profiles · routes swap lanes each storey</text>'
        )
        for lane in (0, 1):
            x0 = left + lane * (span / 2 + 10)
            length = span / 2 - 20
            pts = []
            for t, z in ((0, 0), (0.4, 0.5), (0.6, 0.5), (1, 1)):
                if lane:
                    z = 1 - z
                pts.append(f"{x0 + t * length:g},{sy + sh - z * sh:g}")
            out.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1645ff" stroke-width="2"/>'
            )
            out.append(
                f'<text x="{x0:g}" y="{sy + sh + 16:g}" font-size="9">Lane {lane + 1}: {"0 → 1.5 → 3" if lane == 0 else "3 → 1.5 → 0"} m</text>'
            )
        out.append(
            f'<text x="20" y="{sy + sh + 37:g}" font-size="9">Mid-landings are separate; stacked flights: {plan["clearStackedHeadroom"]:.2f} m vertical clearance (simplified).</text>'
        )
    out.extend(
        [
            f'<text x="20" y="{vh - 65:g}" font-size="10">Flights {plan["clearFlightWidth"]:.2f} m · landings {plan["landingDepth"]:.2f} m · treads {plan["tread"]:.2f} m</text>',
            f'<text x="20" y="{vh - 47:g}" font-size="10">{plan["riserCount"]} risers × {plan["riserHeight"] * 1000:.1f} mm = {plan["floorHeight"]:.2f} m storey</text>',
            f'<text x="20" y="{vh - 29:g}" font-size="9" fill="#64748b">Green = openings · pale outer band = grid rounding · {CORE_LAYOUT_VERSION}</text>',
            f'<text x="20" y="{vh - 12:g}" font-size="9" fill="#64748b">Experimental geometry; no fire, accessibility or lift-capacity certification.</text></g></svg>',
        ]
    )
    return "\n".join(out) + "\n"
