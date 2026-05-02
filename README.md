🤖 LeRobot Pick-and-Place (ACT Policy)

This project demonstrates a robotic pick-and-place task using the LeRobot framework and an ACT (Action Chunking Transformer) policy.
Instead of relying on unstable live inference, the final demo uses dataset replay, ensuring consistent and reliable execution.

🎯 Demo Overview

Task: Pick and place object using SO101 robotic arm
Method: Behavior cloning with ACT policy
Input: Camera + robot state
Output: Joint actions
Final Demo: Replay of recorded successful episodes

🚀 Key Features

End-to-end pipeline:
Data collection (teleoperation)
Dataset creation
Policy training (ACT)
Evaluation & replay
Stable demo via:
Pre-recorded successful episodes
Deterministic replay (no inference noise)

📁 Project Structure
├── data/
│   ├── pick_place/          # Training dataset
│   └── eval_pick_place/     # Evaluation runs
├── outputs/
│   └── act_pick_place/      # Trained model checkpoints
├── scripts/
│   └── (custom helpers if any)
├── run_policy_trained_model.py
└── README.md


⚙️ Setup

1. Clone repository
git clone <your-repo-url>
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

Requires Python 3.10+ and CUDA (for GPU inference)

🎮 Usage

▶️ Replay a successful episode (Recommended Demo)
lerobot-replay \
  --dataset.root=./data/pick_place \
  --episode-index=0 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=Grip_follower

📌 Change --episode-index to test different runs.

📊 View dataset (optional)
python viewer.py

(Displays recorded camera frames for a specific episode)

🧠 Model Details

Architecture: ACT (Action Chunking Transformer)
Backbone: ResNet18
Input:
RGB image (640x480)
Robot state (joint positions)
Output:
Sequence of joint actions

⚠️ Notes

Live policy inference may be unstable due to:
distribution shift
real-world noise
timing constraints
For demonstration purposes, replay is used to ensure consistency

🏁 Results

Successful pick-and-place sequences recorded
Smooth and repeatable execution via replay
End-to-end robotics pipeline completed

🙌 Acknowledgements

HuggingFace LeRobot
ACT Policy (Action Chunking Transformer)
Open-source robotics community

📌 Future Work

Improve generalization of policy
Robust real-time inference
Multi-object manipulation

📬 Contact

For questions or collaboration, feel free to reach out.
For greater details in files like google drive link is given below  
https://drive.google.com/drive/folders/1IfawpfWOUnokMD5u0isglrUXuR6QBH4E?usp=drive_link

