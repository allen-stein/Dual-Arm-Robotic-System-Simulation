#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# Arm parameters (link lengths)
l1 = 1.0  # Length of the first link for each arm
l2 = 1.0  # Length of the second link for each arm
arm_distance = 2.0  # Distance between the base of the two arms

# Initial position for the arms (can be modified)
theta1_left_init = np.pi / 2  # Initial joint angle 1 for the left arm
theta2_left_init = np.pi / 2  # Initial joint angle 2 for the left arm
theta1_right_init = np.pi / 4  # Initial joint angle 1 for the right arm
theta2_right_init = np.pi / 4  # Initial joint angle 2 for the right arm
time_to_target = 3.0  # Time to reach target in seconds
# Target position (to be set by the user)
xd1, yd1 = -1,1.5  # Left arm target position
xd2, yd2 = 1,1.5  # Right arm target position
target_left = [xd1, yd1]
target_right = [xd2, yd2]
previous_target_left = [-100, -100]
previous_target_right = [-100, -100]
theta_dot_previous=[0,0,0,0]

dt=0.1
# Function for forward kinematics
# Assuming mass (m) and inertia (I) are defined for each link:


def newton_euler_with_external_forces(
    theta1_left, theta2_left, theta1_dot_left, theta2_dot_left,
    theta1_ddot_left, theta2_ddot_left,
    theta1_right, theta2_right, theta1_dot_right, theta2_dot_right,
    theta1_ddot_right, theta2_ddot_right,
    F_ext, r_ext
):


    mass1_left = 100.0  # Mass of left arm's link 1
    mass2_left = 100.0  # Mass of left arm's link 2
    mass1_right = 100.0  # Mass of right arm's link 1
    mass2_right = 100.0  # Mass of right arm's link 2

    # Inertia (I) for the links (this is a simplified assumption, a more complex model might include 3D inertia tensors)
    I1_left = 1  # Inertia of left arm's link 1
    I2_left = 1  # Inertia of left arm's link 2
    I1_right = 1  # Inertia of right arm's link 1
    I2_right = 1  # Inertia of right arm's link 2

    # Define the center of mass for each link in terms of their lengths
    com1_left = l1 / 2  # Center of mass of left link 1 (half the length)
    com2_left = l2 / 2  # Center of mass of left link 2
    com1_right = l1 / 2  # Center of mass of right link 1
    com2_right = l2 / 2  # Center of mass of right link 2
    # Initialize forces and torques to zero
    F1_left = np.array([0, 0])  # Force on link 1 (left)
    F2_left = np.array([0, 0])  # Force on link 2 (left)

    F1_right = np.array([0, 0])  # Force on link 1 (right)
    F2_right = np.array([0, 0])  # Force on link 2 (right)

    # tau1_left = 0  # Torque on joint 1 (left)
    # tau2_left = 0  # Torque on joint 2 (left)
    #
    # tau1_right = 0  # Torque on joint 1 (right)
    # tau2_right = 0  # Torque on joint 2 (right)

    # Step 1: Backward Recursion to compute forces and torques starting from the end effector

    # Left arm calculations
    # Link 2 (left arm)
    a2_left = np.array([theta2_ddot_left * np.cos(theta1_left + theta2_left),
                        theta2_ddot_left * np.sin(theta1_left + theta2_left)])  # Acceleration at link 2
    F2_left = mass2_left * a2_left  # Force on link 2

    # External force acting on the end effector
    F_end_effector_left = F2_left + F_ext[0]  # Include external force in the force on link 2
    tau2_ext_left = np.cross(r_ext, F_ext[0])  # External torque from the applied force

    # Ensure com2_left and F2_left are 2D vectors
    com2_left = np.array([com2_left, 0])  # Convert to a 2D vector (assuming it is along x-axis)
    #F2_left = np.array([F2_left[0], 0])  # Assuming F2_left is a scalar, convert to 2D force vector along x-axis

    # Correct cross product calculation (position vector x force vector)
    tau2_left = I2_left * theta2_ddot_left + np.cross(com2_left, F2_left) + tau2_ext_left  # Combine internal and external torques

    # Link 1 (left arm)
    a1_left = np.array([theta1_ddot_left * np.cos(theta1_left), theta1_ddot_left * np.sin(theta1_left)])
    F1_left = mass1_left * a1_left  # Force on link 1
    tau1_left = I1_left * theta1_ddot_left + np.cross(np.array([com1_left, 0]), F1_left)  # Combine internal forces and torques

    # Right arm calculations
    # Link 2 (right arm)
    a2_right = np.array([theta2_ddot_right * np.cos(theta1_right + theta2_right),
                         theta2_ddot_right * np.sin(theta1_right + theta2_right)])  # Acceleration at link 2
    F2_right = mass2_right * a2_right  # Force on link 2

    # External force acting on the end effector
    # print("Shape of F2_right:", theta2_ddot_right.shape)
    # print("Shape of F_ext:", F_ext.shape)
    F_end_effector_right = F2_right+F_ext[1]  # Include external force in the force on link 2
    tau2_ext_right = np.cross(r_ext, F_ext[1])  # External torque from the applied force

    # Ensure com2_right and F2_right are 2D vectors
    com2_right = np.array([com2_right, 0])  # Convert to 2D vector
    F2_right = np.array([F2_right[0], 0])  # Convert to 2D force vector

    # Correct cross product calculation for the right arm
    tau2_right = I2_right * theta2_ddot_right + np.cross(com2_right, F2_right) + tau2_ext_right  # Combine internal and external torques

    # Link 1 (right arm)
    a1_right = np.array([theta1_ddot_right * np.cos(theta1_right), theta1_ddot_right * np.sin(theta1_right)])
    F1_right = mass1_right * a1_right  # Force on link 1
    tau1_right = I1_right * theta1_ddot_right + np.cross(np.array([com1_right, 0]), F1_right)  # Combine internal forces and torques

    return F1_left, F2_left, F1_right, F2_right, tau1_left, tau2_left, tau1_right, tau2_right


def rod_endpoints(length, angle, center):
    # Convert angle from degrees to radians using numpy
    angle_rad = np.radians(angle)

    # Unpack center coordinates
    x_center, y_center = center

    # Calculate half length
    half_length = length / 2

    # Calculate the endpoints
    x_left = x_center - half_length * np.cos(angle_rad)
    y_left = y_center - half_length * np.sin(angle_rad)

    x_right = x_center + half_length * np.cos(angle_rad)
    y_right = y_center + half_length * np.sin(angle_rad)

    return x_left, y_left, x_right, y_right


def forward_kinematics_left(theta1, theta2):
    x = -arm_distance / 2 + l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

# Jacobian matrix calculation
def jacobian(theta1, theta2):
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)
    return np.array([[j11, j12], [j21, j22]])

# Inverse kinematics using Newton-Euler method
def inverse_kinematics_left(xd, yd, theta1_init, theta2_init, max_iter=1000, tol=0.03, learning_rate=0.1):
    theta1 = theta1_init
    theta2 = theta2_init

    for i in range(max_iter):
        # Forward kinematics to get the current end effector position
        x, y = forward_kinematics_left(theta1, theta2)

        # Compute error in position
        error = np.array([xd - x, yd - y])
        #print(error)
        #print(xd,x)
        # If the error is small enough, stop
        if np.linalg.norm(error) < tol:
            #print(f"first reach,{np.linalg.norm(error)}")
            print("left reachd")
            break


        # Calculate the Jacobian matrix
        J = jacobian(theta1, theta2)

        # Compute the pseudo-inverse of the Jacobian
        J_inv = np.linalg.pinv(J)

        # Calculate the change in joint angles
        delta_theta = J_inv @ error

        # Update the joint angles using the learning rate
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
        #print(error,delta_theta,theta1,theta2)
        # theta1 += delta_theta[0]
        # theta2 += delta_theta[1]

    return theta1, theta2
# Function for forward kinematics
def forward_kinematics_right(theta1, theta2):
    x = arm_distance / 2 + l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

# Inverse kinematics using Newton-Euler method
def inverse_kinematics_right(xd, yd, theta1_init, theta2_init, max_iter=1000, tol=0.03, learning_rate=0.1):
    theta1 = theta1_init
    theta2 = theta2_init

    for i in range(max_iter):
        # Forward kinematics to get the current end effector position
        x, y = forward_kinematics_right(theta1, theta2)

        # Compute error in position
        error = np.array([xd - x, yd - y])
        #print(error)
        #print(xd,x)
        # If the error is small enough, stop
        if np.linalg.norm(error) < tol:
            #print(f"first reach,{np.linalg.norm(error)}")
            print("right reachd")
            break


        # Calculate the Jacobian matrix
        J = jacobian(theta1, theta2)

        # Compute the pseudo-inverse of the Jacobian
        J_inv = np.linalg.pinv(J)

        # Calculate the change in joint angles
        delta_theta = J_inv @ error

        # Update the joint angles using the learning rate
        theta1 +=delta_theta[0]
        theta2 +=delta_theta[1]
        #print(error,delta_theta,theta1,theta2)
        # theta1 += delta_theta[0]
        # theta2 += delta_theta[1]

    return theta1, theta2

# Animation update function
def update(frame, line1, line2, line3, line4,line6, arm_data, target_left, target_right, current_left, current_right,previous_target_left,previous_target_right,velocity_left,velocity_right,theta_dot_previous,dt):
    # Get current target positions for both arms
    xd1, yd1 = target_left
    xd2, yd2 = target_right

    if target_left != previous_target_left:
        theta1_left, theta2_left = inverse_kinematics_left(target_left[0], target_left[1], current_left[0], current_left[1])
        previous_target_left = target_left  # Update previous target position
        # Compute the velocity based on the time to target and the distance to target
        velocity_left = np.array([theta1_left - current_left[0], theta2_left - current_left[1]]) / time_to_target

    if target_right != previous_target_right:

        theta1_right, theta2_right = inverse_kinematics_right(target_right[0], target_right[1], current_right[0], current_right[1])
        previous_target_right = target_right  # Update previous target position
        velocity_right = np.array([theta1_right - current_right[0], theta2_right - current_right[1]]) / time_to_target
        F_ext = np.array([[0, 0], [0, 0]])  # External force set to zero (2x2 matrix)

        r_ext = np.array([0, 0])  # External force position set to zero
                # Now, use the calculated velocities as joint velocities for the Newton-Euler calculation
        theta1_dot_left = velocity_left[0]  # Velocity of left arm (joint 1)
        # print(theta1_dot_left)
        theta2_dot_left = velocity_left[1]  # Velocity of left arm (joint 2)
        theta1_dot_right = velocity_right[0]  # Velocity of right arm (joint 1)
        theta2_dot_right = velocity_right[1]  # Velocity of right arm (joint 2)

        theta1_dot_previous = theta_dot_previous[0]
        theta2_dot_previous = theta_dot_previous[1]
        theta1_dot_previous_right = theta_dot_previous[2]
        theta2_dot_previous_right = theta_dot_previous[3]

        theta1_ddot_left = (theta1_dot_left - theta1_dot_previous) / dt
        theta2_ddot_left = (theta2_dot_left - theta2_dot_previous) / dt
        theta1_ddot_right = (theta1_dot_right - theta1_dot_previous_right) / dt
        theta2_ddot_right = (theta2_dot_right - theta2_dot_previous_right) / dt
        # print("Shape of theta2_ddot_right:", theta2_ddot_right.shape)
        # print("Shape of theta2_dot_right:", theta2_dot_right.shape)
        # #print("Shape of theta2_dot_previous_right:", theta2_dot_previous_right.shape)
        # print(theta_dot_previous)

        # Store the previous velocities for the next time step
        theta1_dot_previous = theta1_dot_left
        theta2_dot_previous = theta2_dot_left
        theta1_dot_previous_right = theta1_dot_right
        theta2_dot_previous_right = theta2_dot_right
        theta_dot_previous=[theta1_dot_previous,theta2_dot_previous,theta1_dot_previous_right,theta2_dot_previous_right]


        F1_left, F2_left, F1_right, F2_right, tau1_left, tau2_left, tau1_right, tau2_right = newton_euler_with_external_forces(
            current_left[0], current_left[1], theta1_dot_left, theta2_dot_left, theta1_ddot_left, theta2_ddot_left,
            current_right[0], current_right[1], theta1_dot_right, theta2_dot_right, theta1_ddot_right, theta2_ddot_right,
            F_ext, r_ext)
    # Print or use forces and torques in your animation
    # print("Forces on left arm:", F1_left, F2_left)
    # print("Torques on joints (left):", tau1_left, tau2_left)
    # print("Forces on right arm:", F1_right, F2_right)
    # print("Torques on joints (right):", tau1_right, tau2_right)

    # Assuming `F1_left`, `F2_left`, `F1_right`, `F2_right` are the forces with x and y components
    F1_left_x, F1_left_y = F1_left  # Extract x and y components of F1 on the left arm
    F2_left_x, F2_left_y = F2_left  # Extract x and y components of F2 on the left arm
    F1_right_x, F1_right_y = F1_right  # Extract x and y components of F1 on the right arm
    F2_right_x, F2_right_y = F2_right  # Extract x and y components of F2 on the right arm

    # Add the forces (components) to the lists
    force_left_1_x.append(F1_left_x)  # Total force in X direction on the left arm
    force_left_2_x.append(F2_left_x)  # Total force in X direction on the left arm
    force_left_1_y.append(F1_left_y)  # Total force in Y direction on the left arm
    force_left_2_y.append(F2_left_y)  # Total force in Y direction on the left arm
    force_right_1_x.append(F1_right_x)  # Total force in X direction on the right arm
    force_right_2_x.append(F2_right_x)  # Total force in X direction on the right arm
    force_right_1_y.append(F1_right_y)  # Total force in Y direction on the right arm
    force_right_2_y.append(F2_right_y)  # Total force in Y direction on the right arm

    # Assuming torque components are already available as scalars
    torques_left_1.append(tau1_left)  # Total torque on the left arm
    torques_left_2.append(tau2_left)  # Total torque on the left arm
    torques_right_1.append(tau1_right)  # Total torque on the right arm
    torques_right_2.append(tau2_right)  # Total torque on the right arm

    # Update the force and torque plots with new data
    force_left_1_x_line.set_data(range(len(force_left_1_x)), force_left_1_x)
    force_left_2_x_line.set_data(range(len(force_left_2_x)), force_left_2_x)
    force_left_1_y_line.set_data(range(len(force_left_1_y)), force_left_1_y)
    force_left_2_y_line.set_data(range(len(force_left_2_y)), force_left_2_y)
    force_right_1_x_line.set_data(range(len(force_right_1_x)), force_right_1_x)
    force_right_2_x_line.set_data(range(len(force_right_2_x)), force_right_2_x)
    force_right_1_y_line.set_data(range(len(force_right_1_y)), force_right_1_y)
    force_right_2_y_line.set_data(range(len(force_right_2_y)), force_right_2_y)

    tau_left_1_line.set_data(range(len(torques_left_1)), torques_left_1)
    tau_left_2_line.set_data(range(len(torques_left_2)), torques_left_2)
    tau_right_1_line.set_data(range(len(torques_right_1)), torques_right_1)
    tau_right_2_line.set_data(range(len(torques_right_2)), torques_right_2)


    current_left[0] += velocity_left[0] * dt
    current_left[1] += velocity_left[1] * dt
    current_right[0] += velocity_right[0] * dt
    current_right[1] += velocity_right[1] * dt
    theta1_left=current_left[0]
    theta2_left=current_left[1]
    theta1_right=current_right[0]
    theta2_right=current_right[1]

    # # Calculate positions based on forward kinematics
    x1_left, y1_left = -arm_distance / 2 + l1 * np.cos(theta1_left), l1 * np.sin(theta1_left)
    x2_left, y2_left = x1_left + l2 * np.cos(theta1_left + theta2_left), y1_left + l2 * np.sin(theta1_left + theta2_left)
    #x1_right, y1_right = arm_distance / 2 - l1 * np.cos(theta1_right), l1 * np.sin(theta1_right)

    x1_right, y1_right = arm_distance / 2 + l1 * np.cos(theta1_right), l1 * np.sin(theta1_right)
    x2_right, y2_right = x1_right + l2 * np.cos(theta1_right + theta2_right), y1_right + l2 * np.sin(theta1_right + theta2_right)

    # length_left_link1 = np.sqrt((x1_left+1)**2 + y1_left**2)  # Length of the first link of the left arm
    # length_left_link2 = np.sqrt((x2_left - x1_left)**2 + (y2_left - y1_left)**2)  # Length of the second link of the left arm
    #
    # length_right_link1 = np.sqrt((x1_right-1)**2 + y1_right**2)  # Length of the first link of the right arm
    # length_right_link2 = np.sqrt((x2_right - x1_right)**2 + (y2_right - y1_right)**2)  # Length of the second link of the right arm

    #print(f"L1,L2,R1,R2: {length_left_link1},{length_left_link2},{length_right_link1},{length_right_link2}")
    #print(f"Left arm Link 2 length: {length_left_link2}")
    #print(f"Right arm Link 1 length: {length_right_link1}")
    #print(f"Right arm Link 2 length: {length_right_link2}")
    # Update the arm plot with the new positions
    arm_data[0].set_data([-arm_distance / 2, x1_left], [0, y1_left])  # Left arm Link 1
    arm_data[1].set_data([x1_left, x2_left], [y1_left, y2_left])  # Left arm Link 2

    arm_data[2].set_data([arm_distance / 2, x1_right], [0, y1_right])  # Right arm Link 1
    arm_data[3].set_data([x1_right, x2_right], [y1_right, y2_right])  # Right arm Link 2
    line1.set_data([-arm_distance / 2, x1_left], [0, y1_left])  # Left arm Link 1
    line2.set_data([x1_left, x2_left], [y1_left, y2_left])  # Left arm Link 2

    line3.set_data([arm_distance / 2, x1_right], [0, y1_right])  # Right arm Link 1
    line4.set_data([x1_right, x2_right], [y1_right, y2_right])  # Right arm Link 2
    line6.set_data([xd1, xd2], [yd1, yd2])
    # Update current positions for both arms

    # print(f"Left arm angles: {theta1_left}, {theta2_left}")
    # print(np.linalg.norm([xd1 - x2_left, yd1 - y2_left]),np.linalg.norm([xd2 - x2_right, yd2 - y2_right]))
    # print("target,end->left,right")
    # print([xd1 - x2_left, yd1 - y2_left],[xd2 - x2_right, yd2 - y2_right])
    # print(xd1,x2_left,xd2,x2_right)

    # Check if target has been reached, then prompt for new target positions
    if np.linalg.norm([xd1 - x2_left, yd1 - y2_left]) < 0.03 and np.linalg.norm([xd2 - x2_right, yd2 - y2_right]) < 0.03:
        # Get new target positions from the user
        print("Target reached. Enter new target positions:")
        length, angle, center_x,center_y = get_user_input("ROD")

        target_left[0], target_left[1],target_right[0], target_right[1]=rod_endpoints(length, angle,(center_x,center_y))

    #return arm_data
    #return line1, line2, line3, line4, line6
    previous_target_left = [-100, -100]
    previous_target_right = [-100, -100]
    return target_left,target_right,current_left,current_right,previous_target_left,previous_target_right,theta_dot_previous,velocity_left,velocity_right


# Get user input for new target position (when the arm reaches the target)
def get_user_input(target_name):
    x = float(input(f"Enter new length for {target_name} arm: "))
    y = float(input(f"Enter new angle for {target_name} arm: "))
    z = float(input(f"Enter new x-center_x for {target_name} arm: "))
    w = float(input(f"Enter new y-center_y for {target_name} arm: "))
    return x, y,z,w

# Set up the figure for animation


# Initialize the figure for animation with two subplots
fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8,6))



ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('2-Arm Robotic System Animation')

# Initial arm lines (empty)
line1, = ax.plot([], [], 'ro-', label='Left Arm Link 1')  # Left arm Link 1 (red)
line2, = ax.plot([], [], 'go-', label='Left Arm Link 2')  # Left arm Link 2 (green)

line3, = ax.plot([], [], 'bo-', label='Right Arm Link 1')  # Right arm Link 1 (blue)
line4, = ax.plot([], [], 'yo-', label='Right Arm Link 2')  # Right arm Link 2 (yellow)
line5, = ax.plot([-arm_distance / 2, arm_distance / 2], [0, 0], 'ko-', label='Base Connection')
line6, = ax.plot([], [], 'mo--', label="Target Line")
ax.legend()


# Arm plot
# ax1.set_xlim(0, 100)
# ax1.set_ylim(-1, 1)
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_title('Force Components (X and Y) for Left Arm')
force_left_1_x_line, = ax1.plot([], [], label='Left Arm Force 1 X')
force_left_1_y_line, = ax1.plot([], [], label='Left Arm Force 1 Y')
force_left_2_x_line, = ax1.plot([], [], label='Left Arm Force 2 X')
force_left_2_y_line, = ax1.plot([], [], label='Left Arm Force 2 Y')
ax1.legend()

# Force plot (Left and Right Forces in X and Y directions)
force_right_1_x_line, = ax2.plot([], [], label='Right Arm Force 1 X')
force_right_1_y_line, = ax2.plot([], [], label='Right Arm Force 1 y')
force_right_2_x_line, = ax2.plot([], [], label='Right Arm Force 2 X')
force_right_2_y_line, = ax2.plot([], [], label='Right Arm Force 2 Y')
# ax2.set_xlim(0, 100)
# ax2.set_ylim(-10, 10)
ax2.set_xlabel('Frames')
ax2.set_ylabel('Force (N)')
ax2.set_title('Force Components (X and Y) for Right Arm')
ax2.legend()


# Torque plot (Left and Right torques)
tau_left_1_line, = ax3.plot([], [], label='Left Arm 1 Torque')
tau_left_2_line, = ax3.plot([], [], label='Left Arm 2 Torque')
tau_right_1_line, = ax3.plot([], [], label='Right Arm 1 Torque')
tau_right_2_line, = ax3.plot([], [], label='Right Arm 2 Torque')
# ax3.set_xlim(0, 100)
# ax3.set_ylim(-1, 1)
ax3.set_xlabel('Frames')
ax3.set_ylabel('Torque (Nm)')
ax3.set_title('Torque Plot')
ax3.legend()

# Torque plot (Left and Right torques)
# tau_left_line, = ax4.plot([], [], label='Left Arm Torque')
# tau_right_line, = ax4.plot([], [], label='Right Arm Torque')
# ax4.set_xlim(0, 100)
# ax4.set_ylim(-10, 10)
# ax4.set_xlabel('Frames')
# ax4.set_ylabel('Torque (Nm)')
# ax4.set_title('Torque Plot')
# ax4.legend()

# List of arm data (links to update in the animation)
arm_data = [line1, line2, line3, line4]

# Set initial arm positions
current_left = [theta1_left_init, theta2_left_init]
current_right = [theta1_right_init, theta2_right_init]

# Set initial target positions

theta1_left, theta2_left = inverse_kinematics_left(xd1, yd1, current_left[0], current_left[1])
theta1_right, theta2_right = inverse_kinematics_right(xd2, yd2, current_right[0], current_right[1])
velocity_left = np.array([theta1_left - current_left[0], theta2_left - current_left[1]]) / time_to_target
velocity_right = np.array([theta1_right - current_right[0], theta2_right - current_right[1]]) / time_to_target

# Initialize the force and torque lists as empty lists
force_left_1_x = []  # Total force in X direction on the left arm
force_left_2_x = []  # Total force in X direction on the left arm
force_left_1_y = []  # Total force in Y direction on the left arm
force_left_2_y = []  # Total force in Y direction on the left arm
force_right_1_x = []  # Total force in X direction on the right arm
force_right_2_x = []  # Total force in X direction on the right arm
force_right_1_y = []  # Total force in Y direction on the right arm
force_right_2_y = []  # Total force in Y direction on the right arm

# Assuming torque components are already available as scalars
torques_left_1 = []  # Total torque on the left arm
torques_left_2 = []  # Total torque on the left arm
torques_right_1 = []  # Total torque on the right arm
torques_right_2 = []  # Total torque on the right arm


# Set up animation
def animate_func(frame):
    global target_left, target_right, current_left, current_right,previous_target_left,previous_target_right,velocity_left,velocity_right,theta_dot_previous


    #print(current_left,current_right)
    # Update the arm positions for the first frame

    if frame == 0:
        # Set initial position for both arms
        target_left,target_right,current_left,current_right,previous_target_left,previous_target_right,theta_dot_previous,velocity_left,velocity_right=update(0, line1, line2, line3, line4, line6,arm_data, target_left, target_right, current_left, current_right,previous_target_left,previous_target_right,velocity_left,velocity_right,theta_dot_previous,dt)
    #print(current_left,current_right)
    # Call the update function for each subsequent frame
    target_left,target_right,current_left,current_right,previous_target_left,previous_target_right,theta_dot_previous,velocity_left,velocity_right= update(frame, line1, line2, line3, line4,line6, arm_data, target_left, target_right, current_left, current_right,previous_target_left,previous_target_right,velocity_left,velocity_right,theta_dot_previous,dt)
    ax2.relim()  # Recalculate limits
    ax2.autoscale_view()  # Automatically adjust the view limits
    ax1.relim()  # Recalculate limits
    ax1.autoscale_view()  # Automatically adjust the view limits
    ax3.relim()  # Recalculate limits
    ax3.autoscale_view()  # Automatically adjust the view limits
    # print("enterrer")
    # print(target_right,previous_target_right)
    # print("hihihih")
    return



# Set up animation
ani = animation.FuncAnimation(fig, animate_func, frames=100, interval=100)
# Show the animation
plt.grid(True)
plt.legend(loc='upper left')

plt.show()
