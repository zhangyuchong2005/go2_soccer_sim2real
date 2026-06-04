from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["numpy", "torch", "mujoco"]:
        print(f"{name}: {bool(importlib.util.find_spec(name))}")
    import mujoco

    xml = ROOT / "assets" / "quadruped_soccer_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    print(f"mujoco_scene: nq={model.nq} nv={model.nv} nu={model.nu}")


if __name__ == "__main__":
    main()
