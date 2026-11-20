"""Metric service-core geometry for the ML environment, not a regulatory model.

One source defines components, raster reservations, API data and SVG drawings.
Height bands and lift counts are experimental priors, not traffic/egress design.
"""
from __future__ import annotations
from dataclasses import dataclass
from html import escape
import math

CORE_LAYOUT_VERSION = 'compact-core.v2'

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

    def __post_init__(self):
        values=(self.floor_height,self.tread,self.flight_width,self.landing_depth,self.wall,self.shaft_width,self.shaft_depth,self.car_width,self.car_depth,self.door_width)
        if any(not math.isfinite(v) or v<=0 for v in values) or self.risers_per_flight<2 or self.stair_gap<0:
            raise ValueError('Core geometry must be finite and positive')
        if self.landing_depth < self.flight_width:
            raise ValueError('Landing depth must accommodate a flight width')
        if self.shaft_width-2*self.wall<self.car_width or self.shaft_depth-2*self.wall<self.car_depth:
            raise ValueError('Lift car must fit inside the shaft walls')
        if self.door_width>min(self.car_width,self.flight_width):
            raise ValueError('Door must fit the car and stair')

@dataclass(frozen=True)
class HeightBand:
    tier: str
    minimum: int
    maximum: int
    stairs: int
    lifts: int

DEFAULT_HEIGHT_BANDS=(HeightBand('T1',1,5,1,0),HeightBand('T2',6,10,1,1),
    HeightBand('T3',11,18,1,2),HeightBand('T4',19,30,2,3),HeightBand('T5',31,60,2,4))

def make_plan(band: HeightBand, p: CorePlanParameters | None = None) -> dict:
    p=p or CorePlanParameters()
    if band.stairs not in (1,2) or band.lifts<0 or band.minimum<1 or band.maximum<band.minimum:
        raise ValueError('Unsupported height band')
    run=(p.risers_per_flight-1)*p.tread
    sw=2*p.wall+2*p.flight_width+p.stair_gap
    length=2*p.wall+2*p.landing_depth+run
    lift_bank=band.lifts*p.shaft_width
    width=band.stairs*sw+lift_bank
    zones=[];parts=[];doors=[]
    def rect(kind,x,y,w,h,label,**extra):
        return dict(type=kind,x=round(x,6),y=round(y,6),w=round(w,6),h=round(h,6),label=label,**extra)
    stair_xs=[0.] if band.stairs==1 else [0.,sw+lift_bank]
    for i,x in enumerate(stair_xs):
        zones.append(rect('stair',x,0,sw,length,f'Stair {i+1}'))
        parts.append(rect('landing',x+p.wall,p.wall,sw-2*p.wall,p.landing_depth,'Floor landing',stair=i))
        parts.append(rect('landing',x+p.wall,p.wall+p.landing_depth+run,sw-2*p.wall,p.landing_depth,'Half landing',stair=i))
        for flight in (0,1):
            parts.append(rect('flight',x+p.wall+flight*(p.flight_width+p.stair_gap),p.wall+p.landing_depth,
                p.flight_width,run,f'Flight {flight+1}',stair=i,risers=p.risers_per_flight,treads=p.risers_per_flight-1,direction=1 if flight==0 else -1))
        # Side opening joins the floor landing directly to the adjacent lobby.
        if band.lifts:
            doors.append(dict(type='stair_door',x=round(x+sw if i==0 else x,6),y=round(p.wall+.15,6),width=.9,orientation='vertical'))
        else:
            doors.append(dict(type='entrance',x=round(x+.95,6),y=0.,width=.9,orientation='horizontal'))
    if band.lifts:
        lobby_depth=length-p.shaft_depth
        zones.append(rect('lobby',sw,0,lift_bank,lobby_depth,'Shared lobby'))
        doors.append(dict(type='entrance',x=round(sw+(lift_bank-.9)/2,6),y=0.,width=.9,orientation='horizontal'))
        for i in range(band.lifts):
            x=sw+i*p.shaft_width;y=lobby_depth
            zones.append(rect('lift',x,y,p.shaft_width,p.shaft_depth,f'Lift {i+1}'))
            parts.append(rect('car',x+(p.shaft_width-p.car_width)/2,y+p.wall+.30,p.car_width,p.car_depth,f'Car {i+1}'))
            doors.append(dict(type='lift_door',x=round(x+(p.shaft_width-p.door_width)/2,6),y=round(y,6),width=p.door_width,orientation='horizontal'))
    width=round(width,6);length=round(length,6)
    return dict(version=CORE_LAYOUT_VERSION,tier=band.tier,minHeight=band.minimum,maxHeight=band.maximum,
        stairCount=band.stairs,liftCount=band.lifts,metricWidth=width,metricLength=length,
        metricArea=round(width*length,4),width=math.ceil(width-1e-9),length=math.ceil(length-1e-9),
        floorHeight=p.floor_height,riserCount=2*p.risers_per_flight,riserHeight=p.floor_height/(2*p.risers_per_flight),tread=p.tread,
        clearFlightWidth=p.flight_width,landingDepth=p.landing_depth,wallThickness=p.wall,
        shaftClearWidth=p.shaft_width-2*p.wall,shaftClearDepth=p.shaft_depth-2*p.wall,
        carWidth=p.car_width,carDepth=p.car_depth,zones=zones,components=parts,doors=doors,
        assumptions='Research layout prior; lift counts depend on this configured height schedule. Not a code, traffic or evacuation assessment.')

def plan_library():
    return [make_plan(b) for b in DEFAULT_HEIGHT_BANDS]

def render_plan_svg(plan: dict) -> str:
    """Scaled vector plan with separate flights, landings, car/shaft and openings."""
    scale=50.;ox=70.;oy=85.;w=plan['metricWidth'];h=plan['metricLength']
    vw=max(390.,ox+w*scale+80);vh=oy+h*scale+130
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:g} {vh:g}" role="img" aria-label="{plan["tier"]} core plan">',
         '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#1645ff"/></marker></defs>',
         f'<rect width="{vw:g}" height="{vh:g}" fill="white"/>',
         f'<g font-family="Arial,sans-serif" fill="#182238"><text x="20" y="25" font-size="16" font-weight="bold">{plan["tier"]} · {plan["minHeight"]}–{plan["maxHeight"]} floors</text>',
         f'<text x="20" y="46" font-size="11">{plan["stairCount"]} stair(s) · {plan["liftCount"]} lift(s) · {plan["metricArea"]:g} m² metric / {plan["width"]*plan["length"]} m² grid reservation</text>']
    def line(x1,y1,x2,y2,stroke='#344158',width=1,extra=''):
        out.append(f'<line x1="{ox+x1*scale:g}" y1="{oy+y1*scale:g}" x2="{ox+x2*scale:g}" y2="{oy+y2*scale:g}" stroke="{stroke}" stroke-width="{width}" {extra}/>')
    def rect(c,fill,stroke='#344158',width=1):
        out.append(f'<rect x="{ox+c["x"]*scale:g}" y="{oy+c["y"]*scale:g}" width="{c["w"]*scale:g}" height="{c["h"]*scale:g}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    # Dashed outline shows the conservative one-metre solver reservation.
    rect(dict(x=0,y=0,w=plan['width'],h=plan['length']),'#fafbff','#b8c7ef')
    for z in plan['zones']:
        rect(z,{'stair':'#e2e8f0','lobby':'#fff5d8','lift':'#c7d2fe'}[z['type']],width=2)
        if z['type'] in ('stair','lift'):
            t=plan['wallThickness'];rect(dict(x=z['x']+t,y=z['y']+t,w=z['w']-2*t,h=z['h']-2*t),'white')
        elif z['type']=='lobby':
            out.append(f'<text x="{ox+(z["x"]+z["w"]*.5)*scale:g}" y="{oy+(z["y"]+z["h"]*.55)*scale:g}" text-anchor="middle" font-size="11">Lobby</text>')
    for c in plan['components']:
        if c['type']=='landing':
            rect(c,'#f0f6ff','#94a3b8')
            out.append(f'<text x="{ox+(c["x"]+c["w"]*.5)*scale:g}" y="{oy+(c["y"]+c["h"]*.55)*scale:g}" text-anchor="middle" font-size="8">{escape(c["label"])}</text>')
        elif c['type']=='flight':
            rect(c,'white','#94a3b8')
            for j in range(c['treads']+1):
                y=c['y']+c['h']*j/c['treads'];line(c['x'],y,c['x']+c['w'],y,stroke='#64748b',width=.8)
            mid=c['x']+c['w']/2
            y0=c['y']+.2;y1=c['y']+c['h']-.2
            if c['direction']<0:y0,y1=y1,y0
            line(mid,y0,mid,y1,stroke='#1645ff',width=1.8,extra='marker-end="url(#arrow)"')
            out.append(f'<text x="{ox+mid*scale:g}" y="{oy+(c["y"]+c["h"]*.55)*scale:g}" font-size="8" fill="#1645ff">UP</text>')
        elif c['type']=='car':rect(c,'#eef2ff','#4338ca',1.3)
    for door in plan['doors']:
        x=door['x'];y=door['y'];d=door['width']
        if door['orientation']=='horizontal':
            line(x,y,x+d,y,stroke='white',width=plan['wallThickness']*scale*2+2);line(x,y,x+d,y,stroke='#16a34a',width=2)
            if door['type']=='entrance':line(x,y,x,y-.7,stroke='#344158',width=1)
        else:
            line(x,y,x,y+d,stroke='white',width=plan['wallThickness']*scale*2+2);line(x,y,x,y+d,stroke='#16a34a',width=2)
    # Metric dimensions rather than inferring scale from a rectangle label.
    line(0,-.3,w,-.3);line(-.3,0,-.3,h)
    out.append(f'<text x="{ox+w*scale/2:g}" y="{oy-23:g}" text-anchor="middle" font-size="11">{w:.2f} m</text>')
    out.append(f'<text transform="translate({ox-26:g},{oy+h*scale/2:g}) rotate(-90)" text-anchor="middle" font-size="11">{h:.2f} m</text>')
    out.extend([f'<text x="20" y="{vh-65:g}" font-size="10">Flights {plan["clearFlightWidth"]:.2f} m · landings {plan["landingDepth"]:.2f} m · treads {plan["tread"]:.2f} m</text>',
        f'<text x="20" y="{vh-47:g}" font-size="10">{plan["riserCount"]} risers × {plan["riserHeight"]*1000:.1f} mm = {plan["floorHeight"]:.2f} m storey</text>',
        f'<text x="20" y="{vh-29:g}" font-size="9" fill="#64748b">Green = openings · pale outer band = grid rounding · {CORE_LAYOUT_VERSION}</text>',
        f'<text x="20" y="{vh-12:g}" font-size="9" fill="#64748b">Experimental geometry; no fire, accessibility or lift-capacity certification.</text></g></svg>'])
    return '\n'.join(out)+'\n'
