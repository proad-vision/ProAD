# perlin.py
import torch
import math
import numpy as np


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """
    Generates 2D Perlin noise using NumPy.
    Corrected to handle arbitrary shapes and ensure correct grid dimensions.
    Includes commented-out debug prints and an optional epsilon nudge for diagnostics.
    """
    # --- Grid and Index Calculation ---
    # Absolute coordinates for each pixel in the res-grid
    pixel_i_coords, pixel_j_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    abs_fx = pixel_i_coords.astype(np.float64) * res[0] / shape[0]
    abs_fy = pixel_j_coords.astype(np.float64) * res[1] / shape[1]

    # --- Optional Epsilon Nudge (for diagnosing boundary artifacts) ---
    # If you uncomment this, ensure this is the *only* source for abs_fx, abs_fy
    # that then feed into x_indices, y_indices, and grid calculation.
    # epsilon_nudge = 1e-7 # Small positive offset
    # abs_fx = abs_fx + epsilon_nudge
    # abs_fy = abs_fy + epsilon_nudge
    # --- End of Epsilon Nudge ---

    # Integer indices from absolute coordinates
    x_indices = np.floor(abs_fx).astype(np.int32)
    y_indices = np.floor(abs_fy).astype(np.int32)

    # Fractional coordinates (grid) from absolute coordinates
    grid_x_frac = abs_fx % 1.0
    grid_y_frac = abs_fy % 1.0
    grid = np.stack((grid_x_frac, grid_y_frac), axis=-1)

    # --- Debug Prints (uncomment to use if artifacts persist) ---
    # print(f"rand_perlin_2d_np: shape={shape}, res={res}")
    # print(f"rand_perlin_2d_np: abs_fx min/max: {abs_fx.min()}, {abs_fx.max()}")
    # print(f"rand_perlin_2d_np: grid_x_frac min/max: {grid_x_frac.min()}, {grid_x_frac.max()}")
    # print(f"rand_perlin_2d_np: x_indices min/max: {x_indices.min()}, {x_indices.max()}")
    # --- End of Debug Prints ---

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Ensure indices for gradients are within bounds
    # Max x_indices will be res[0]-1 if abs_fx can reach res[0]-epsilon
    # So x_indices+1 can be res[0]. gradients array is (res[0]+1, ...) so index res[0] is valid.
    x_indices_p1 = np.clip(x_indices + 1, 0, res[0])  # Safeguard, though direct indexing should be fine
    y_indices_p1 = np.clip(y_indices + 1, 0, res[1])  # Safeguard

    grad_00 = gradients[x_indices, y_indices]
    grad_10 = gradients[x_indices_p1, y_indices]
    grad_01 = gradients[x_indices, y_indices_p1]
    grad_11 = gradients[x_indices_p1, y_indices_p1]

    dot = lambda grad_field, shift_vec: (
            np.stack(
                (grid[..., 0] + shift_vec[0], grid[..., 1] + shift_vec[1]),
                axis=-1
            ) * grad_field
    ).sum(axis=-1)

    n00 = dot(grad_00, [0, 0])
    n10 = dot(grad_10, [-1, 0])
    n01 = dot(grad_01, [0, -1])
    n11 = dot(grad_11, [-1, -1])

    t_fade = fade(grid)

    n0 = lerp_np(n00, n10, t_fade[..., 0])
    n1 = lerp_np(n01, n11, t_fade[..., 0])

    final_noise = math.sqrt(2) * lerp_np(n0, n1, t_fade[..., 1])
    return final_noise


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape, dtype=np.float64)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        current_res_x = max(1, int(round(frequency * res[0])))  # round before int
        current_res_y = max(1, int(round(frequency * res[1])))

        noise += amplitude * generate_perlin_noise_2d(shape, (current_res_x, current_res_y))
        frequency *= 2.0
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res, fade_func=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """
    Generates 2D Perlin noise using NumPy. Updated for robustness.
    """
    pixel_i_coords, pixel_j_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    # --- Epsilon Nudge (Optional, for diagnosing boundary artifacts) ---
    # epsilon_nudge = 1e-7
    # abs_fx = pixel_i_coords.astype(np.float64) * res[0] / shape[0] + epsilon_nudge
    # abs_fy = pixel_j_coords.astype(np.float64) * res[1] / shape[1] + epsilon_nudge
    # --- Default path without nudge ---
    abs_fx = pixel_i_coords.astype(np.float64) * res[0] / shape[0]
    abs_fy = pixel_j_coords.astype(np.float64) * res[1] / shape[1]
    # --- End Nudge Section ---

    x_indices = np.floor(abs_fx).astype(np.int32)
    y_indices = np.floor(abs_fy).astype(np.int32)

    grid_x_frac = abs_fx % 1.0
    grid_y_frac = abs_fy % 1.0
    grid = np.stack((grid_x_frac, grid_y_frac), axis=-1)

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    x_indices_p1 = np.clip(x_indices + 1, 0, res[0])
    y_indices_p1 = np.clip(y_indices + 1, 0, res[1])

    grad_00 = gradients[x_indices, y_indices]
    grad_10 = gradients[x_indices_p1, y_indices]
    grad_01 = gradients[x_indices, y_indices_p1]
    grad_11 = gradients[x_indices_p1, y_indices_p1]

    dot = lambda grad_field, shift_vec: (
            np.stack(
                (grid[..., 0] + shift_vec[0], grid[..., 1] + shift_vec[1]),
                axis=-1
            ) * grad_field
    ).sum(axis=-1)

    n00 = dot(grad_00, [0, 0])
    n10 = dot(grad_10, [-1, 0])
    n01 = dot(grad_01, [0, -1])
    n11 = dot(grad_11, [-1, -1])

    t_fade = fade_func(grid)

    n0 = lerp_np(n00, n10, t_fade[..., 0])
    n1 = lerp_np(n01, n11, t_fade[..., 0])

    return math.sqrt(2) * lerp_np(n0, n1, t_fade[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """
    Generates 2D Perlin noise using PyTorch. Updated for robustness.
    """
    default_device = torch.device('cpu')  # For simplicity, or pass device as argument

    pixel_i_pt, pixel_j_pt = torch.meshgrid(
        torch.arange(shape[0], device=default_device, dtype=torch.long),
        torch.arange(shape[1], device=default_device, dtype=torch.long),
        indexing='ij'
    )

    # --- Epsilon Nudge (Optional, for diagnosing boundary artifacts in PyTorch version) ---
    # epsilon_nudge_pt = 1e-7
    # abs_fx_pt = pixel_i_pt.to(torch.float64) * res[0] / shape[0] + epsilon_nudge_pt if shape[0] > 0 else torch.empty_like(pixel_i_pt, dtype=torch.float64)
    # abs_fy_pt = pixel_j_pt.to(torch.float64) * res[1] / shape[1] + epsilon_nudge_pt if shape[1] > 0 else torch.empty_like(pixel_j_pt, dtype=torch.float64)
    # --- Default path without nudge ---
    abs_fx_pt = pixel_i_pt.to(torch.float64) * res[0] / shape[0] if shape[0] > 0 else torch.empty_like(pixel_i_pt,
                                                                                                       dtype=torch.float64)
    abs_fy_pt = pixel_j_pt.to(torch.float64) * res[1] / shape[1] if shape[1] > 0 else torch.empty_like(pixel_j_pt,
                                                                                                       dtype=torch.float64)
    # --- End Nudge Section ---

    x_indices_pt = torch.floor(abs_fx_pt).long()
    y_indices_pt = torch.floor(abs_fy_pt).long()

    grid_x_frac_pt = abs_fx_pt % 1.0
    grid_y_frac_pt = abs_fy_pt % 1.0
    grid = torch.stack((grid_x_frac_pt, grid_y_frac_pt), dim=-1)

    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, dtype=grid.dtype, device=grid.device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    # Clamping indices to be safe for gradient array access
    x_indices_pt_clamped = torch.clamp(x_indices_pt, 0, res[0] - 1 if res[0] > 0 else 0)  # Max base index is res[0]-1
    y_indices_pt_clamped = torch.clamp(y_indices_pt, 0, res[1] - 1 if res[1] > 0 else 0)
    x_indices_p1_pt_clamped = torch.clamp(x_indices_pt + 1, 0, res[0])  # Max index for +1 is res[0]
    y_indices_p1_pt_clamped = torch.clamp(y_indices_pt + 1, 0, res[1])

    grad_00_pt = gradients[x_indices_pt_clamped, y_indices_pt_clamped]
    grad_10_pt = gradients[x_indices_p1_pt_clamped, y_indices_pt_clamped]
    grad_01_pt = gradients[x_indices_pt_clamped, y_indices_p1_pt_clamped]
    grad_11_pt = gradients[x_indices_p1_pt_clamped, y_indices_p1_pt_clamped]

    dot_pt = lambda grad_field, shift_vec: (
            torch.stack(
                (grid[..., 0] + shift_vec[0], grid[..., 1] + shift_vec[1]),
                dim=-1
            ) * grad_field
    ).sum(dim=-1)

    # Ensure shift_vec is a tensor of the correct dtype and device
    shift_00 = torch.tensor([0, 0], dtype=grid.dtype, device=grid.device)
    shift_10 = torch.tensor([-1, 0], dtype=grid.dtype, device=grid.device)
    shift_01 = torch.tensor([0, -1], dtype=grid.dtype, device=grid.device)
    shift_11 = torch.tensor([-1, -1], dtype=grid.dtype, device=grid.device)

    n00 = dot_pt(grad_00_pt, shift_00)
    n10 = dot_pt(grad_10_pt, shift_10)
    n01 = dot_pt(grad_01_pt, shift_01)
    n11 = dot_pt(grad_11_pt, shift_11)

    t_fade = fade(grid)

    n0 = torch.lerp(n00, n10, t_fade[..., 0])
    n1 = torch.lerp(n01, n11, t_fade[..., 0])

    return math.sqrt(2) * torch.lerp(n0, n1, t_fade[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    default_device = torch.device('cpu')
    noise = torch.zeros(shape, dtype=torch.float64, device=default_device)

    frequency = 1.0
    amplitude = 1.0

    for _ in range(octaves):
        current_res_x = max(1, int(round(frequency * res[0])))  # round before int
        current_res_y = max(1, int(round(frequency * res[1])))

        octave_noise = rand_perlin_2d(shape, (current_res_x, current_res_y))
        noise += amplitude * octave_noise.to(noise.dtype).to(noise.device)

        frequency *= 2.0
        amplitude *= persistence
    return noise