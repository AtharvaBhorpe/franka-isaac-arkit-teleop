

# Project Plan: Imitation Learning with FlexiTac on Franka Emika in Isaac Sim

## 1. Problem Statement
### 1.1 Objective
Develop an end-to-end simulation and learning pipeline to train a **Franka Emika Panda** manipulator equipped with open-source **FlexiTac** piezoresistive tactile sensors to perform dextrous manipulation tasks. The system must learn directly from human demonstrations (imitation learning) using a modified Action Chunking with Transformers (**ACT**) policy within the **LeRobot** ecosystem.

### 1.2 Core Challenges
* **Multi-Modal Alignment:** Merging high-frequency tactile arrays (contact maps) with standard vision tokens (RGB cameras) and proprioceptive states (7-DoF joint positions + gripper state) inside a unified transformer backbone.
* **Sim-to-Real Domain Gap:** Modeling the highly non-linear, soft-body deformation of FlexiTac's three-layer laminate stack (FPC-Velostat-FPC) inside NVIDIA Isaac Sim accurately enough that the trained model can transition to real hardware.
* **LeRobot Integration:** Injecting custom observation channels into the structured Hugging Face LeRobot format without disrupting the native data logging, serialization, and training APIs.

---

## 2. Proposed Architecture & System Design

+-------------------------------------------------------------------------+
| NVIDIA ISAAC SIM |
| |
| +--------------------+ +--------------------+ +---------------+ |
| | Franka Emika Robot | | FlexiTac Gripper | | Environment | |
| | (Proprioception) | | (Contact Penalty) | | (Cameras) | |
| +---------+----------+ +---------+----------+ +-------+-------+ |
+------------|-------------------------|-----------------------|----------+
| | |
v v v
+-------------------------------------------------------------------------+
| DATA SYNCHRONIZATION PIPELINE |
| |
| * Joint States (30Hz) * Tactile Arrays (100Hz+) * RGB Frames (30Hz)|
| * Frequency Downsampling / Timestamp Alignment to Camera Frame Rate |
+-------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------+
| LeFlexiTac DATASET ENGINE |
| |
| Formats observations into Hugging Face Parquet + MP4 structures: |
| - observation.state - observation.images.main |
| - observation.tactile (New Modality) |
+-------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------+
| LeFlexiTac TRAINER (ACT) |
| |
| [Vision Tokens] + [Tactile Tokens] + [Proprioception Tokens] |
| | |
| v |
| Modified CVAE Transformer Encoder |
+-------------------------------------------------------------------------+


---

## 3. Step-by-Step Implementation Approach

### Phase 1: Digital Twin Assembly (Isaac Sim)
1. **Robot Asset:** Load the default, instanceable Franka Emika Panda USD (`/Isaac/Robots/Franka/franka_instanceable.usd`).
2. **Gripper Integration:** Clear the default `panda_hand` links from the Stage Tree. Import the FlexiTac-modified gripper USD from the official `FlexiTac-IsaacSim-Simulation` asset suite. Create a rigid/fixed joint between `panda_link8` and the root link of the new gripper.
3. **Sensor Physics Calibration:** Configure the penalty-based contact model on the sensor pads. Adjust the simulated contact stiffness and damping coefficients to approximate the soft-body compression of the physical Velostat material.

### Phase 2: Data Collection & Alignment
1. **Simulation Stepping:** Run the Omniverse physics loop at a step size matching the highest necessary frequency (e.g., $1/100$s or $1/120$s).
2. **Temporal Alignment:** Implement a telemetry script within Isaac Sim's Python API to sample robot joint positions (`observation.state`), RGB camera feeds (`observation.images.main`), and tactile matrices. Downsample or interpolate the high-frequency tactile arrays to seamlessly lock step with the $30\text{ Hz}$ camera frame rate.
3. **Dataset Packaging:** Leverage the **LeFlexiTac** serialization scripts to pipe the synchronized episodic demonstrations into the standardized LeRobot storage format. Verify that `observation.tactile` stores clean, normalized 2D matrices representing contact pressure patterns.

### Phase 3: Model Adaptation (ACT + LeFlexiTac)
1. **Codebase Setup:** Clone and configure the `LeFlexiTac` ecosystem (the specialized fork modifying Hugging Face's LeRobot).
2. **Policy Tokenization:** Adapt the ACT architecture configuration (`policy=act`). The tactile array must be treated like a miniature image patch or flattened numerical tensor, projected through a linear layer into a common embedding space, and appended as sequence tokens into the core Transformer Encoder alongside vision and state features.
3. **Training Execution:** Launch the training routine using a multi-modal configuration structure:
   ```bash
   python train.py policy=act env=franka_flexitac features=[state, images, tactile]
   ```

### Phase 4: Sim-to-Real Mitigation
1. **Domain Randomization:** Inject deliberate synthetic noise, scale scaling factors, and apply random spatial shifting to the simulated tactile arrays during dataset preprocessing to prevent the neural network from overfitting to specific simulator-specific contact boundaries.
2. **Friction Variations:** Dynamically vary object friction parameters inside Isaac Sim across different demonstration episodes to ensure the trained ACT policy relies on sensory feedback rather than hardcoded kinematic paths.

---

## 4. Evaluation Metrics
* **Task Success Rate:** Percentage of completed pick-and-place or insertion tasks over 50 evaluation runs in simulation.
* **Grip Force Optimization:** Peak and average contact force applied during manipulation, ensuring the sensor prevents object slippage without executing excessive force (crushing).
* **Inference Latency:** Time delay introduced by tokenizing and embedding the additional tactile arrays within the ACT transformer chunking cycle (targeting $< 20\text{ ms}$).

---

To tailor this project document further, let me know:
* What **specific manipulation task** (e.g., picking up fragile objects, peg-in-hole insertion) are you targeting?
* Are you planning to map the tactile data as a **2D pressure image** or a **flattened raw vector** of taxel values? 
* Would you like me to draft an example **Isaac Sim Python script** for extracting and synchronizing these multi-modal states?
