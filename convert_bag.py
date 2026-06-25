import numpy as np
from pathlib import Path
from tqdm import tqdm

# Rosbags libraries for handling the .db3 file without a live ROS environment
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# LeRobot library for formatting the dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- CONFIGURATION ---
# Points out of lerobot_ws and into your spot workspace recordings
BAG_PATH = Path("../spot_ros2_multi_ws/src/recordings/bag_20260622_180946") 
DATASET_REPO_ID = "spot_teleop_dataset"
FPS = 30  # Target frequency for policy training

def main():
    # 1. Create the structured LeRobot Dataset template
    # Adjust the shape (e.g., 12 for Spot's joints) to match what you recorded!
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features={
            "observation.state": {"dtype": "float32", "shape": (12,)},  # Current joint positions
            "action": {"dtype": "float32", "shape": (12,)},             # Target joint commands
        }
    )

    print(f"Opening ROS 2 bag at: {BAG_PATH.resolve()}")
    
    # 2. Open and parse the bag
    with Reader(BAG_PATH) as reader:
        for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Processing Bag"):
            
            # Parse state observations (e.g., standard sensor_msgs/msg/JointState)
            if connection.topic == "/joint_states":
                msg = deserialize_cdr(rawdata, connection.msgtype)
                
                current_state = np.array(msg.position, dtype=np.float32)
                # Map this to your action data stream as needed
                action = np.array(msg.position, dtype=np.float32) 

                # 3. Commit this step to your dataset
                dataset.add_frame({
                    "observation.state": current_state,
                    "action": action,
                })

    # 4. Wrap up the recording session and save metadata
    dataset.save_episode(task_description="Spot demonstration run.")
    print(f"\n🎉 Success! LeRobot dataset built at: {dataset.root}")

if __name__ == "__main__":
    main()