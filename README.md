# 7-DOF Franka Panda SE(3) Inverse Kinematics Companion Website

This repository contains the visual companion website and the simulation codebase for the 7-DOF Franka Emika Panda SE(3) Inverse Kinematics (IK) research. It showcases the trajectory playbacks, simulation dashboards, and redundancy resolution comparisons that are impossible to fully convey in the written paper.

---

## Repository Structure

*   **[index.html](file:///Users/keivalya/Desktop/Northeastern%20University/Semester%203/ME5250%20Robot%20Mechanics/mujoco%20example/project-2/franka-se3-ik/index.html)**: Main visual companion webpage.
*   **[style.css](file:///Users/keivalya/Desktop/Northeastern%20University/Semester%203/ME5250%20Robot%20Mechanics/mujoco%20example/project-2/franka-se3-ik/style.css)**: Academic serif stylesheet.
*   **[app.js](file:///Users/keivalya/Desktop/Northeastern%20University/Semester%203/ME5250%20Robot%20Mechanics/mujoco%20example/project-2/franka-se3-ik/app.js)**: Interactive null-space video controller logic.
*   **[assets/](file:///Users/keivalya/Desktop/Northeastern%20University/Semester%203/ME5250%20Robot%20Mechanics/mujoco%20example/project-2/franka-se3-ik/assets)**: Folder containing loopable MP4 videos and pre-rendered performance charts.
*   **[experiments/](file:///Users/keivalya/Desktop/Northeastern%20University/Semester%203/ME5250%20Robot%20Mechanics/mujoco%20example/project-2/franka-se3-ik/experiments)**: Python simulation code for reproducing all numerical experiments, plotting charts, and exporting data.

---

## Reproducing the Experiments

All python scripts to run the experiments are stored in the `/experiments` folder. Check [experiments/README.md](./experiments/README.md) for execution details.

---

## Running the Webpage

*   **Direct Execution**: Open `index.html` directly in any web browser.
*   **Development Server**: Run `npm run dev` to start a local development server.
