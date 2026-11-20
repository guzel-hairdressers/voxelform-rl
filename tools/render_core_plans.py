"""Regenerate inspectable metric SVGs and JSON from the canonical core library."""
import argparse,json
from pathlib import Path
from voxelform.core_plans import plan_library,render_plan_svg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('docs/core-plans'));args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plans=plan_library()
    for plan in plans:(args.output/(plan['tier']+'.svg')).write_text(render_plan_svg(plan))
    (args.output/'layouts.json').write_text(json.dumps(plans,indent=2)+'\n')
    print(args.output.resolve())
if __name__=='__main__':main()
