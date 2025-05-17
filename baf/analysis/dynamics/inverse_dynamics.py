import numpy as np
import pandas as pd


def calculate_joint_moments(kinematics, grf, anthropometry, side='right'):
    """
    Calculate joint moments using inverse dynamics.
    
    This is a simplified implementation of inverse dynamics for calculating
    joint moments during gait. It uses a simplified 2D sagittal plane model.
    
    Parameters
    ----------
    kinematics : pandas.DataFrame
        DataFrame containing joint angles and angular velocities
    grf : pandas.DataFrame
        DataFrame containing ground reaction forces
    anthropometry : dict
        Dictionary containing anthropometric data:
        - mass: body mass in kg
        - height: body height in m
        - segment_lengths: dict with segment lengths in m
        - segment_masses: dict with segment masses in kg
        - segment_coms: dict with segment center of mass locations (fraction of segment length)
    side : str
        Side to calculate moments for ('right' or 'left')
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing joint moments
    """
    # Extract required data
    mass = anthropometry['mass']
    height = anthropometry['height']
    
    # If segment lengths not provided, estimate from height
    if 'segment_lengths' not in anthropometry:
        # Approximate segment lengths as fractions of height
        segment_lengths = {
            'foot': 0.152 * height,
            'shank': 0.246 * height,
            'thigh': 0.245 * height
        }
    else:
        segment_lengths = anthropometry['segment_lengths']
    
    # If segment masses not provided, estimate from total mass
    if 'segment_masses' not in anthropometry:
        # Approximate segment masses as fractions of total mass
        segment_masses = {
            'foot': 0.0145 * mass,
            'shank': 0.0465 * mass,
            'thigh': 0.1 * mass
        }
    else:
        segment_masses = anthropometry['segment_masses']
    
    # If segment COMs not provided, use typical values
    if 'segment_coms' not in anthropometry:
        # Approximate COM locations as fractions of segment length from proximal end
        segment_coms = {
            'foot': 0.5,  # Simplified as 50% of foot length
            'shank': 0.433,  # 43.3% of shank length from knee
            'thigh': 0.433  # 43.3% of thigh length from hip
        }
    else:
        segment_coms = anthropometry['segment_coms']
    
    # Create empty dataframe for moments
    moments = pd.DataFrame(index=kinematics.index)
    
    # Calculate moments (simplified 2D sagittal plane model)
    # This is a simplified implementation that doesn't account for full 3D dynamics
    
    # Gravity
    g = 9.81  # m/s^2
    
    # Extract GRF data
    if 'vertical_force' in grf.columns:
        fz = grf['vertical_force'].values
    else:
        fz = np.zeros(len(kinematics))
    
    if 'anterior_posterior_force' in grf.columns:
        fy = grf['anterior_posterior_force'].values
    else:
        fy = np.zeros(len(kinematics))
    
    # Simplified ankle moment calculation
    # M_ankle = F_GRF_y * lever_arm_y + F_GRF_z * lever_arm_z
    # Assuming GRF application point is at the forefoot (2/3 of foot length from ankle)
    lever_arm_y = segment_lengths['foot'] * 0.67
    moments['ankle_moment'] = -fy * lever_arm_y
    
    # Simplified knee moment calculation
    # M_knee = M_ankle + F_GRF_y * lever_arm_y + F_GRF_z * lever_arm_z + m_shank * g * COM_shank
    shank_com_distance = segment_lengths['shank'] * segment_coms['shank']
    shank_gravity_moment = segment_masses['shank'] * g * shank_com_distance
    moments['knee_moment'] = moments['ankle_moment'] - fy * segment_lengths['shank'] - shank_gravity_moment
    
    # Simplified hip moment calculation
    # M_hip = M_knee + F_GRF_y * lever_arm_y + F_GRF_z * lever_arm_z + m_thigh * g * COM_thigh
    thigh_com_distance = segment_lengths['thigh'] * segment_coms['thigh']
    thigh_gravity_moment = segment_masses['thigh'] * g * thigh_com_distance
    moments['hip_moment'] = moments['knee_moment'] - fy * segment_lengths['thigh'] - thigh_gravity_moment
    
    # Convert to appropriate units and sign convention
    # Positive = extension, Negative = flexion
    moments = moments * -1  # Flip sign for clinical convention
    
    return moments


def calculate_joint_powers(kinematics, moments):
    """
    Calculate joint powers from joint angular velocities and moments.
    
    Parameters
    ----------
    kinematics : pandas.DataFrame
        DataFrame containing joint angles and angular velocities
    moments : pandas.DataFrame
        DataFrame containing joint moments
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing joint powers
    """
    # Create empty dataframe for powers
    powers = pd.DataFrame(index=kinematics.index)
    
    # Check if angular velocities are provided
    if 'ankle_angular_velocity' not in kinematics.columns:
        # Calculate angular velocities using central difference
        dt = 0.01  # Assuming 100 Hz sampling rate
        if 'ankle_angle' in kinematics.columns:
            kinematics['ankle_angular_velocity'] = np.gradient(kinematics['ankle_angle'].values, dt)
        if 'knee_angle' in kinematics.columns:
            kinematics['knee_angular_velocity'] = np.gradient(kinematics['knee_angle'].values, dt)
        if 'hip_angle' in kinematics.columns:
            kinematics['hip_angular_velocity'] = np.gradient(kinematics['hip_angle'].values, dt)
    
    # Calculate powers (P = M * ω)
    if 'ankle_moment' in moments.columns and 'ankle_angular_velocity' in kinematics.columns:
        powers['ankle_power'] = moments['ankle_moment'] * kinematics['ankle_angular_velocity']
    
    if 'knee_moment' in moments.columns and 'knee_angular_velocity' in kinematics.columns:
        powers['knee_power'] = moments['knee_moment'] * kinematics['knee_angular_velocity']
    
    if 'hip_moment' in moments.columns and 'hip_angular_velocity' in kinematics.columns:
        powers['hip_power'] = moments['hip_moment'] * kinematics['hip_angular_velocity']
    
    return powers


def calculate_joint_work(powers, time_col='time'):
    """
    Calculate joint work by integrating power over time.
    
    Parameters
    ----------
    powers : pandas.DataFrame
        DataFrame containing joint powers
    time_col : str
        Name of column containing time data
        
    Returns
    -------
    dict
        Dictionary containing positive, negative, and net work for each joint
    """
    work = {}
    
    # Get time data
    if time_col in powers.columns:
        time = powers[time_col].values
    else:
        # Assume uniform sampling
        time = np.linspace(0, 1, len(powers))
    
    # Calculate time step
    dt = np.diff(time)
    if len(dt) > 0:
        dt = np.append(dt, dt[-1])
    else:
        dt = np.ones(len(time)) * 0.01  # Default to 0.01s if time data is invalid
    
    # Calculate work for each joint
    for joint in ['ankle', 'knee', 'hip']:
        power_col = f'{joint}_power'
        if power_col in powers.columns:
            # Integrate power to get work
            power_data = powers[power_col].values
            work_data = np.cumsum(power_data * dt)
            
            # Calculate positive, negative, and net work
            positive_work = np.sum(np.maximum(power_data, 0) * dt)
            negative_work = np.sum(np.minimum(power_data, 0) * dt)
            net_work = work_data[-1]
            
            work[joint] = {
                'positive_work': positive_work,
                'negative_work': negative_work,
                'net_work': net_work
            }
    
    return work 