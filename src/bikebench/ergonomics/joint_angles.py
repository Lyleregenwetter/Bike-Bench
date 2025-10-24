import numpy as np
import torch
from scipy.stats import norm

USE_DICT = {
    "road": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (45, 5),
        "opt_awrist_angle": (90, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
    "mtb": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (50, 5),
        "opt_awrist_angle": (90, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
    "commute": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (52, 5),
        "opt_awrist_angle": (85, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
}


# #####
# Bike Ergonomic Angle Fit Calculator
# Calculates body angles given bike and body measurements
# Calculates ergonomic score/probability of fit given use case
######


#############################################
# Masking Functions for Triangle Inequality #
#############################################

def deg_to_r(deg: torch.Tensor) -> torch.Tensor:
    """
    Converts degrees to radians (works with tensors or scalars).
    """
    return deg.to(dtype=torch.float32) * (torch.pi / 180)


def law_of_cosines(a, b, c):
    return (a**2 + b**2 - c**2) / (2 * a * b)

def rad_to_d(rad: torch.Tensor) -> torch.Tensor:
    """
    Converts radians to degrees (works with tensors or scalars).
    """
    return rad.to(dtype=torch.float32) * (180 / torch.pi)

###################################
# FUNCTIONS FOR CALCUATING ANGLES #
###################################
def min_knee_angle(bike_vectors, body_vectors, apply_penalty, eps=1e-6):
    """
    Input:
        bike vector, body vector
        np array bike vector:
            [SX, SY, HX, HY, CL]^T
            (seat_x, seat_y, hbar_x, hbar_y, crank len)
            Origin is bottom bracket
        np array body vector:
            [LL, UL, TL, AL, FL, AA]
            (lowleg, upleg, torso len, arm len, foot len, ankle angle)"""
    
    UL = body_vectors[:, 0:1]
    LL = body_vectors[:, 1:2]
    AL = body_vectors[:, 2:3]
    TL = body_vectors[:, 3:4]
    FL = body_vectors[:, 4:5]
    AA = deg_to_r(body_vectors[:, 6:7])
    EA = deg_to_r(180 - body_vectors[:, 7:8])

    HX = bike_vectors[:, 0:1]  # Hand x
    HY = bike_vectors[:, 1:2]  # Hand y
    SX = bike_vectors[:, 2:3] * -1  # Hip x (flip because convention here is positive x is forward on the bike)
    SY = bike_vectors[:, 3:4]  # Hip y
    CL = bike_vectors[:, 4:5]  # Crank length

    CA = torch.atan2(SY, SX)  # Crank angle in radians

    
    # distance to pedal in furthest position 
    LX = SX + CL * torch.cos(CA) 
    LY = SY + CL * torch.sin(CA)

    # Law of cosines for ankle
    k2t = torch.sqrt(LL ** 2 + FL ** 2 - 2 * LL * FL * torch.cos(AA)) #knee to toe distance
    

    h2t = torch.sqrt(LX ** 2 + LY ** 2) #hip to toe distance
    alpha_hkt_cos = law_of_cosines(k2t, UL, h2t) #hip knee toe angle cosine
    alpha_hkt_cos_clamped = (torch.clamp(alpha_hkt_cos, -1.0 + eps, 1.0 - eps))
    alpha_hkt = torch.arccos(alpha_hkt_cos_clamped)

    alpha_akt_cos = law_of_cosines(k2t, LL, FL)  # knee ankle toe angle cosine
    alpha_akt = torch.arccos(alpha_akt_cos)

    ke = torch.pi - alpha_hkt + alpha_akt  # knee extension angle in radians

    if apply_penalty:
        #NOTE These corrections are only valid if akt is smaller than the target knee angle, which should almost always be the case
        case1 = h2t > UL + k2t  # results in 180 degrees for hip knee toe angle, results in too small knee extension angle
        case1_correction = - (h2t - UL - k2t)

        case2 = UL > h2t + k2t  # results in 0 degrees for hip knee toe angle, results in too large knee extension angle
        case2_correction = UL - h2t - k2t

        case3 = k2t > h2t + UL  # results in 0 degrees for hip knee toe angle, results in too large knee extension angle
        case3_correction = k2t - h2t - UL

        ke = ke + case1 * case1_correction + case2 * case2_correction + case3 * case3_correction

    return rad_to_d(ke)

def back_armpit_angles(bike_vectors, body_vectors, apply_penalty, eps=1e-6):
    """
    Input: bike_vector, body_vector
    Output: back angle, armpit to elbow angle, armpit to wrist angle in degrees

    np array bike vector:
            [SX, SY, HX, HY, CL]^T
    np array body vector:
            [LL, UL, TL, AL, FL, AA]
    """
    UL = body_vectors[:, 0:1]
    LL = body_vectors[:, 1:2]
    AL = body_vectors[:, 2:3]
    TL = body_vectors[:, 3:4]
    FL = body_vectors[:, 4:5]
    AA = deg_to_r(body_vectors[:, 6:7])
    EA = deg_to_r(180 - body_vectors[:, 7:8])

    HX = bike_vectors[:, 0:1]  # Hand x
    HY = bike_vectors[:, 1:2]  # Hand y
    SX = bike_vectors[:, 2:3] * -1  # Hip x (flip because convention here is positive x is forward on the bike)
    SY = bike_vectors[:, 3:4]  # Hip y
    CL = bike_vectors[:, 4:5]  # Crank length
    SY = SY + 0.06 # in BikeCAD, "Hip" reference point for upper torso is offset from from lower torso by 60 mm
    
    #saddle to handle measurements
    sth_dx = HX - SX
    sth_dy = HY - SY
    sth_dist = torch.sqrt(sth_dx**2 + sth_dy**2)
    sth_ang = torch.atan2(sth_dy, sth_dx)

    # Law of cosines to simulate elbow bend
    shoulder_to_hand = torch.sqrt((AL / 2) ** 2 + (AL / 2) ** 2 - 2 * (AL / 2) * (AL / 2) * torch.cos(EA))


    tors_angle_cos = law_of_cosines(sth_dist, TL, shoulder_to_hand)

    tors_angle__cos_clamped = torch.clamp(tors_angle_cos, -1.0 + eps, 1.0 - eps)
    tors_ang = torch.arccos(tors_angle__cos_clamped)

    shoulder_angle_cos = law_of_cosines(shoulder_to_hand, TL, sth_dist)
    shoulder_angle_cos_clamped = torch.clamp(shoulder_angle_cos, -1.0 + eps, 1.0 - eps)
    shoulder_ang = torch.arccos(shoulder_angle_cos_clamped)

    if apply_penalty:
        case1 = shoulder_to_hand > sth_dist + TL #results in 180 degrees for back angle and 0 degrees for shoulder angle
        case1_correction_back = shoulder_to_hand - sth_dist - TL
        case1_correction_shoulder = -(shoulder_to_hand - sth_dist - TL)

        case2 = sth_dist> TL + shoulder_to_hand #results in 0 degrees for back angle and 180 degrees for shoulder angle
        case2_correction_back = - (sth_dist - TL - shoulder_to_hand)
        case2_correction_shoulder = sth_dist - TL - shoulder_to_hand

        case3 = TL > shoulder_to_hand + sth_dist #results in 0 degrees for back angle and 0 degrees for shoulder angle
        case3_correction_back = - (TL - shoulder_to_hand - sth_dist)
        case3_correction_shoulder = - (TL - shoulder_to_hand - sth_dist)

        tors_ang = tors_ang + case1*case1_correction_back + case2*case2_correction_back + case3*case3_correction_back
        shoulder_ang = shoulder_ang + case1*case1_correction_shoulder + case2*case2_correction_shoulder + case3*case3_correction_shoulder

    back_angle = tors_ang + sth_ang


    return rad_to_d(back_angle), rad_to_d(shoulder_ang)


def constraints(bike_vectors, body_vectors, eps=1e-6):
    """
    Input: bike_vector, body_vector
    Output: back angle, armpit to elbow angle, armpit to wrist angle in degrees

    np array bike vector:
            [SX, SY, HX, HY, CL]^T
    np array body vector:
            [LL, UL, TL, AL, FL, AA]
    """

    N = body_vectors.shape[0]
    device = body_vectors.device

    # Append default ankle (90 deg) and elbow (20 deg)
    body_vectors = torch.cat((
        body_vectors,
        torch.full((N, 1), 90.0, device=device),
        torch.full((N, 1), 20.0, device=device)
    ), dim=1)

    UL = body_vectors[:, 0:1]
    LL = body_vectors[:, 1:2]
    AL = body_vectors[:, 2:3]
    TL = body_vectors[:, 3:4]
    FL = body_vectors[:, 4:5]
    AA = deg_to_r(body_vectors[:, 6:7])
    EA = deg_to_r(180 - body_vectors[:, 7:8])

    HX = bike_vectors[:, 0:1]  # Hand x
    HY = bike_vectors[:, 1:2]  # Hand y
    SX = bike_vectors[:, 2:3] * -1  # Hip x (flip because convention here is positive x is forward on the bike)
    SY = bike_vectors[:, 3:4]  # Hip y
    CL = bike_vectors[:, 4:5]  # Crank length

    #saddle to handle measurements
    sth_dx = HX - SX
    sth_dy = HY - SY
    sth_dist = torch.sqrt(sth_dx**2 + sth_dy**2)

    # Law of cosines to simulate elbow bend
    shoulder_to_hand = torch.sqrt((AL / 2) ** 2 + (AL / 2) ** 2 - 2 * (AL / 2) * (AL / 2) * torch.cos(EA))

    arm_too_long = shoulder_to_hand - sth_dist - TL
    saddle_too_far_from_handle = sth_dist - TL - shoulder_to_hand
    torso_too_long = TL - shoulder_to_hand - sth_dist

    CA = torch.atan2(SY, SX)  # Crank angle in radians

    # distance to pedal in furthest position 
    LX = SX + CL * torch.cos(CA) 
    LY = SY + CL * torch.sin(CA)

    # Law of cosines for ankle
    k2t = torch.sqrt(LL ** 2 + FL ** 2 - 2 * LL * FL * torch.cos(AA)) #knee to toe distance
    h2t = torch.sqrt(LX ** 2 + LY ** 2) #hip to toe distance

    saddle_to_toe_too_long = h2t - UL - k2t
    upper_leg_too_long = UL - h2t - k2t
    knee_too_toe_too_long = k2t - h2t - UL

    all_constraints = torch.concat([arm_too_long, saddle_too_far_from_handle, torso_too_long, saddle_to_toe_too_long, upper_leg_too_long, knee_too_toe_too_long], dim=1)
    return all_constraints


def all_angles(bike_vectors, body_vectors, apply_penalty):
    """
    Input: bike, body, arm angle (at elbow) in degrees
    Output: tuple (min_ke angle, back angle, awrist angle) in degrees
    """
    N = body_vectors.shape[0]
    device = body_vectors.device

    # Append default ankle (90 deg) and elbow (20 deg)
    body_vectors = torch.cat((
        body_vectors,
        torch.full((N, 1), 90.0, device=device),
        torch.full((N, 1), 20.0, device=device)
    ), dim=1)

    ke_ang = min_knee_angle(bike_vectors, body_vectors, apply_penalty)

    b_angs, aw_angs = back_armpit_angles(bike_vectors, body_vectors, apply_penalty)

    return torch.cat((ke_ang, b_angs, aw_angs), dim=1)


############################
# FUNCTIONS FOR SCORING #
############################

def adjusted_nll(bike_vectors: torch.Tensor, body_vectors: torch.Tensor, use_vec: list[str], apply_penalty = False) -> torch.Tensor:
    """
    Adjusted NLL = log(pdf at optimal) - log(pdf at observed).
    This ensures that the minimum is zero at the optimal angle.
    Output: (N, 3) torch tensor: [knee_nll, back_nll, awrist_nll]
    """
    angles = all_angles(bike_vectors, body_vectors, apply_penalty)  # (N, 3)

    means = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][0], USE_DICT[u]["opt_back_angle"][0], USE_DICT[u]["opt_awrist_angle"][0]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    stds = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][1], USE_DICT[u]["opt_back_angle"][1], USE_DICT[u]["opt_awrist_angle"][1]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    # Log-PDF at observed values
    log_pdf_actual = -((angles - means) ** 2) / (2 * stds ** 2) - torch.log(stds * torch.sqrt(torch.tensor(2 * torch.pi, device=angles.device)))

    # Log-PDF at the optimal (i.e., mean), which is the peak of the Gaussian
    log_pdf_optimal = - torch.log(stds * torch.sqrt(torch.tensor(2 * torch.pi, device=angles.device)))

    # Difference gives adjusted NLL
    adjusted_nll = log_pdf_optimal - log_pdf_actual  # = 0 at the peak
    return adjusted_nll

def dist_to_1SD(bike_vectors: torch.Tensor, body_vectors: torch.Tensor, use_vec: list[str], apply_penalty = False) -> torch.Tensor:
    """
    Distance to 1 standard deviation from the mean for each angle.
    Output: (N, 3) torch tensor: [knee_dist, back_dist, awrist_dist]
    """

    angles = all_angles(bike_vectors, body_vectors, apply_penalty)  # (N, 3)
    means = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][0], USE_DICT[u]["opt_back_angle"][0], USE_DICT[u]["opt_awrist_angle"][0]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    stds = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][1], USE_DICT[u]["opt_back_angle"][1], USE_DICT[u]["opt_awrist_angle"][1]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    # Distance to 1 standard deviation from the mean
    dist_to_1SD = torch.abs(angles - means) - stds
    dist_to_1SD = torch.clamp(dist_to_1SD, min=0)  # Ensure non-negative distances
    return dist_to_1SD

def smooth_piecewise_loss(bike_vectors: torch.Tensor, body_vectors: torch.Tensor, use_vec: list[str], power=10.0) -> torch.Tensor:
    """
    Smooth piecewise loss function for ergonomic angles.
    Function is z^2/2 for angles with z score under 1 SD and z/2 for angles with z score greater than 1 SD.
    Output: (N, 3) torch tensor: [knee_loss, back_loss, awrist_loss]
    """
    angles = all_angles(bike_vectors, body_vectors)  # (N, 3)

    means = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][0], USE_DICT[u]["opt_back_angle"][0], USE_DICT[u]["opt_awrist_angle"][0]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    stds = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][1], USE_DICT[u]["opt_back_angle"][1], USE_DICT[u]["opt_awrist_angle"][1]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    # Smooth piecewise loss
    zscore = torch.abs(angles - means) / stds
    power_mask = zscore < 1.0
    linear_mask = ~power_mask
    power_loss = zscore**power/power
    linear_loss = zscore - 1/power
    loss = power_loss* power_mask + linear_loss * linear_mask
    return loss


