# Dual-Arm Robotic System Simulation

This repository contains a Python-based simulation of a **dual-arm, 2-link robotic system**. It uses inverse kinematics, Newton-Euler dynamics, and real-time animation to simulate robotic motion, visualize forces and torques, and respond to user-defined target positions.



## 🛠 Features

- Dual 2-DOF robotic arms with configurable base spacing.
- Forward and inverse kinematics using Jacobian-based control.
- Newton-Euler dynamic modeling (joint torques and endpoint forces).
- Live `matplotlib` animation of arm motion and force/torque plotting.
- Interactive target input: specify rod endpoints using length, angle, and center.
- Real-time control loop with adjustable time step.

## 📦 Requirements

- Python 3.x
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
