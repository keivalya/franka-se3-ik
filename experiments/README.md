# ME5250 Project 2 &mdash; Trajectory Simulation & Kinematic Reproductions

This directory contains the independent, reproducible Python source code for the inverse kinematics simulation experiments, mathematical charts, and data extractions for the 7-DOF Franka Emika Panda robot.

---

## 🛠️ Installation & Setup

Ensure you have Python 3 installed. Install the required numerical and simulation packages:

```bash
pip install numpy mujoco matplotlib
```

The robot's physical model is loaded from the local [mujoco_menagerie](./mujoco_menagerie/franka_emika_panda/panda.xml) folder.

---

## 🚀 Running the Experiments

### 1. Run tracking simulation & print numerical summary:
```bash
python run_simulation.py
```
*   **What it does**: Runs the 948-waypoint helix trajectory tracking under three solver configurations: Newton-Raphson (no null-space), Newton-Raphson + Null-Space ($\alpha = 0.50$), and Damped Least Squares (DLS).
*   **Outputs**: Prints Cartesian tracking errors, manipulability indexes, condition numbers, and joint limit margins directly to the console. Saves a summary to `simulation_results.json`.

### 2. Plot publication-quality figures:
```bash
python generate_all_figures.py
```
*   **What it does**: Iterates through the helix path and generates all the math charts and diagram figures utilized in the research paper (position error logs, condition numbers, joint limit centering comparison plots).
*   **Outputs**: Generates a local `figures/` folder containing the publication plots (`fig3_fk_verification.png`, `fig4_ik_error.png`, `fig5_manipulability.png`, `fig6_nullspace.png`, etc.).

### 3. Export detailed dataset for the website:
```bash
python export_detailed_data.py
```
*   **What it does**: Simulates the trajectory under all solvers and centering gain settings ($\alpha = 0.00, 0.10, 0.30, 0.50$) and writes the step-by-step joint angle lists and Cartesian translation coordinates.
*   **Outputs**: Generates `simulation_results_detailed.json` which can be wrapped into a JavaScript variable for the paper-style website.
