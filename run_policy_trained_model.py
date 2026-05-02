import torch
import json
from pathlib import Path

from safetensors.torch import load_file

from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower.so_follower import SOFollower


# ---------- CONFIG PATH ----------
ckpt = Path("./outputs/act_pick_place/checkpoints/last/pretrained_model")


# ---------- LOAD DATASET (REQUIRED) ----------
dataset = LeRobotDataset(
    repo_id="vinny/pick_place",
    root="/home/vinny/Documents/Robothon/Local/workspace/data/pick_place"
)


# ---------- LOAD CONFIG ----------
with open(ckpt / "config.json") as f:
    raw_cfg = json.load(f)

# remove invalid key (breaks ACTConfig)
raw_cfg.pop("type", None)


# ---------- FIX FEATURE STRUCTURE ----------
# your version expects attribute access like ft.type
from lerobot.configs.types import FeatureType

class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

def fix_features(feats):
    new_feats = {}

    for k, v in feats.items():
        v = DotDict(v)

        # 🔥 smarter detection based on key name
        if "image" in k or "camera" in k:
            v["type"] = FeatureType.VISUAL

        elif "state" in k or "joint" in k:
            v["type"] = FeatureType.STATE

        elif "action" in k:
            v["type"] = FeatureType.ACTION

        else:
            # fallback (important)
            if v.get("type") == "visual":
                v["type"] = FeatureType.VISUAL
            elif v.get("type") == "state":
                v["type"] = FeatureType.STATE
            elif v.get("type") == "action":
                v["type"] = FeatureType.ACTION

        new_feats[k] = v

    return new_feats


raw_cfg["input_features"] = fix_features(raw_cfg["input_features"])
raw_cfg["output_features"] = fix_features(raw_cfg["output_features"])


# ---------- BUILD CONFIG ----------
cfg = make_policy_config(
    policy_type="act",
    **raw_cfg
)


# ---------- CREATE POLICY ----------
policy = make_policy(
    cfg=cfg,
    ds_meta=dataset.meta   # 🔥 REQUIRED
)


# ---------- LOAD WEIGHTS (SAFE TENSORS) ----------
state_dict = load_file(
    str(ckpt / "model.safetensors"),
    device="cuda"
)

policy.load_state_dict(state_dict)
policy.eval().to("cuda")


# ---------- CONNECT ROBOT ----------
from lerobot.robots.utils import make_robot_from_config

class DotDict(dict):
    def __getattr__(self, name):
        return self.get(name, None)   # 🔥 safer: returns None instead of crashing


robot_cfg = DotDict({
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    "id": "Grip_follower",

    # required defaults
    "calibration_dir": None,
    "use_degrees": True,
    "disable_torque_on_disconnect": True,
    "max_relative_target": None,

    "cameras": {}
})

robot = make_robot_from_config(robot_cfg)
robot.connect()

print("🚀 Running trained ACT policy...")


# ---------- CONTROL LOOP ----------
try:
    while True:
        obs = robot.get_observation()

        with torch.no_grad():
            # try this first
            action = policy.select_action(obs)

        robot.send_action(action)

except KeyboardInterrupt:
    print("\nStopping...")
    robot.disconnect()