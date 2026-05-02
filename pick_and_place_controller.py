import mujoco
import mujoco.viewer
import numpy as np
import os
import cv2

# Base model setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "src/so101_mujoco/mujoco/scene.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# IDs
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i")

# State variables
state = "approach"
step_counter = 0
grasp_counter = 0
descend_counter = 0
release_counter = 0

# Gripper values (OPEN/CLOSE constants based on your XML range)
OPEN = -0.1
CLOSE = 1.4  # Increased for stronger grip

renderer = mujoco.Renderer(model, height=240, width=320)

def move_ee_to_target(model, data, target_pos):
    """
    Stabilized IK Controller to prevent flailing
    """
    ee_pos = data.site_xpos[ee_site_id]
    error = target_pos - ee_pos
    
    # 1. Calculate Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    
    # 2. Slice for arm joints (0-4)
    J = jacp[:, :5]
    
    # 3. Damped Least Squares
    damping = 0.06  # Slightly higher for smoother descend
    f = np.linalg.solve(J @ J.T + damping**2 * np.eye(3), error)
    joint_delta = J.T @ f
    
    # 4. Apply to absolute position targets
    data.ctrl[:5] = data.qpos[:5] + (joint_delta * 0.5)
    
    # 5. Safety Clamp to actuator limits
    for i in range(5):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], ctrl_min, ctrl_max)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        step_counter += 1

        # Current positions
        ee_pos = data.site_xpos[ee_site_id].copy()
        cube_pos = data.xpos[cube_id].copy()
        target_pos = data.xpos[target_id].copy()

        # Waypoints - Lowered grasp height for better contact
        cube_hover = cube_pos + np.array([0, 0, 0.12])
        cube_grasp = cube_pos + np.array([0, 0, 0.045]) # Lowered from 0.05
        target_hover = target_pos + np.array([0, 0, 0.25])
        target_drop = target_pos + np.array([0, 0, 0.17])

        # --- State Machine ---
        
        if state == "approach":
            data.ctrl[5] = OPEN
            move_ee_to_target(model, data, cube_hover)
            if np.linalg.norm(ee_pos - cube_hover) < 0.02:
                state = "descend"

        elif state == "descend":
            data.ctrl[5] = OPEN
            move_ee_to_target(model, data, cube_grasp)
            descend_counter += 1
            if np.linalg.norm(ee_pos - cube_grasp) < 0.012 or descend_counter > 400: 
                state = "grasp"

        elif state == "grasp":
            data.ctrl[5] = CLOSE 
            grasp_counter += 1
            # Increased wait time to ensure physical closure
            if grasp_counter > 250: 
                state = "lift"

        elif state == "lift":
            data.ctrl[5] = CLOSE # Maintain grip
            move_ee_to_target(model, data, cube_hover)
            if np.linalg.norm(ee_pos - cube_hover) < 0.03:
                state = "move"

        elif state == "move":
            data.ctrl[5] = CLOSE # Maintain grip
            move_ee_to_target(model, data, target_hover)
            if np.linalg.norm(ee_pos - target_hover) < 0.02:
                state = "drop"

        elif state == "drop":
            data.ctrl[5] = CLOSE
            move_ee_to_target(model, data, target_drop)
            if np.linalg.norm(ee_pos - target_drop) < 0.02:
                state = "release"

        elif state == "release":
            data.ctrl[5] = OPEN
            release_counter += 1
            if release_counter > 80:
                state = "done"

        # Step Simulation
        mujoco.mj_step(model, data)
        
        # View Updates
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Robot Camera", img)
        cv2.waitKey(1)
        
        if step_counter % 50 == 0:
            print(f"State: {state} | Dist to Target: {np.linalg.norm(ee_pos - cube_grasp):.4f}")
            
        viewer.sync()