import numpy as np
from pathlib import Path
from tqdm import tqdm

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# LeRobot library for formatting the dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- CONFIGURATION ---
BAG_PATH = Path("../spot_ros2_multi_ws/src/recordings/bag_20260622_180946") 
DATASET_REPO_ID = "spot_teleop_dataset"
FPS = 30  

def main():
    # Initialize the dataset layout
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features={
            "observation.state": {"dtype": "float32", "shape": (12,)},  
            "action": {"dtype": "float32", "shape": (12,)},             
        }
    )

    # Set up a standard ROS 2 message definition store
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Opening ROS 2 bag via AnyReader at: {BAG_PATH.resolve()}")
    
    # Read and automatically deserialize messages
    with AnyReader([BAG_PATH], default_typestore=typestore) as reader:
        # Filter connections to just look at our joint states topic
        connections = [x for x in reader.connections if x.topic == "/joint_states"]
        
        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Processing Bag"):

            msg = reader.deserialize(rawdata, connection.msgtype)
            
            current_state = np.array(msg.position, dtype=np.float32)
            action = np.array(msg.position, dtype=np.float32) 

            # Commit frame to your dataset
            dataset.add_frame({
                "observation.state": current_state,
                "action": action,
            })

    # Save metadata
    dataset.save_episode(task_description="Spot demonstration run.")
    print(f"\nLeRobot dataset built at: {dataset.root}")

if __name__ == "__main__":
    main()