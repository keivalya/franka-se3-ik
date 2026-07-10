# Interactive 7-DOF Franka Panda SE(3) IK Website

This repository contains the interactive companion website for the 7-DOF Franka Emika Panda SE(3) Inverse Kinematics (IK) project. It serves as an interactive report detailing redundancy resolution, null-space optimization, and singularity-robust trajectory tracking.

The project simulation core is located in the [me5250-project-2](../me5250-project-2) repository, while this repository builds and presents the web-based interactive interface.

## 🌟 Key Features

1. **Interactive 3D Robot Kinematics Visualizer (Three.js)**:
   - Implements standard/classical Denavit-Hartenberg (DH) forward kinematics dynamically in JavaScript.
   - **Joint Sliders Mode**: Allows manual joint articulation ($q_1 \dots q_7$), showing real-time coordinate frames ($X$-red, $Y$-green, $Z$-blue) and links updated dynamically.
   - **Trajectory Playback Mode**: Animates the robot tracing a helical trajectory, comparing Newton-Raphson, Null-Space joint-limit avoidance, and Damped Least Squares (DLS).
2. **Interactive Null-Space Controller**:
   - Visualizes how the null-space coefficient ($\alpha = 0.00, 0.10, 0.30, 0.50$) pulls joint angles back toward their midpoints to avoid physical limits.
   - Integrates side-by-side MuJoCo simulation MP4 playback with interactive charts plotting mean joint deviations from midpoints.
3. **Synchronized Performance Dashboard (Chart.js)**:
   - **Tracking Error**: High-fidelity line plot showing sub-micron tracking errors for Newton/DLS and transient drifts for active null-space solvers, utilizing logarithmic scaling.
   - **Manipulability & Condition**: Plots manipulability index ($w = \sqrt{\det(J J^T)}$) and condition number ($\kappa$) along the trajectory on dual Y-axes, highlighting singularity boundaries ($w < 0.04$).
   - **Joint Configurations vs. Limits**: Normalizes all 7 joint configurations in the range $[-1.0, 1.0]$, demonstrating when joints approach physical limit boundaries under different solvers.
   - **Scrubber Sync**: Dragging the trajectory scrubber or playing the animation updates the 3D model and draws a synchronized vertical indicator line across all charts.
4. **Kinematic Theory & Mathematics (MathJax)**:
   - Includes standard mathematical formulations of DH transforms, Jacobian pseudoinverses, null-space projection, and Damped Least Squares (DLS) dampening.

## 🛠️ Development & Local Setup

The project uses [Vite](https://vite.dev/) as a fast static file development server.

### Prerequisites

Make sure you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

Install dependencies (Vite) from the root of the `franka-se3-ik` folder:

```bash
npm install
```

### Run Locally

Start the Vite local development server:

```bash
npm run dev
```

Vite will start a local server (typically at `http://localhost:5173`). Open this link in your browser to view the interactive website.

### Production Build

Build the static site into the `dist/` directory for deployment (e.g., GitHub Pages, Vercel, Netlify):

```bash
npm run build
```

This compiles `index.html`, bundles `app.js` and `style.css` (with minification), and copies the `public/assets` folder (containing simulation data and videos) into `dist/assets` as-is.

---
Developed for **ME5250 Robot Mechanics** at Northeastern University.
