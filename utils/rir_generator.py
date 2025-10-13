"""Room Impulse Response (RIR) generation utilities."""

from typing import List, Tuple, Optional
import numpy as np


def generate_rir_image(
    T: int = 512,
    fs: int = 16000,
    room_dim: Tuple[float, float, float] = (10.0, 8.0, 3.0),
    source: Tuple[float, float, float] = (2.0, 3.0, 1.5),
    mic: Tuple[float, float, float] = (5.0, 4.0, 1.5),
    reflection_order: int = 1,
    beta: float = 0.7
) -> np.ndarray:
    """
    Generate a Room Impulse Response (RIR) using a simplified image-source model.
    
    This function implements the image source method for RIR generation in rectangular rooms.
    It considers reflections up to a specified order and applies frequency-independent
    absorption coefficients.

    Args:
        T: Length of the impulse response in samples
        fs: Sampling frequency in Hz
        room_dim: Room dimensions (length, width, height) in meters
        source: Source position (x, y, z) in meters
        mic: Microphone position (x, y, z) in meters
        reflection_order: Maximum order of reflections to consider
        beta: Reflection coefficient (0 < beta < 1), where beta=0 means full absorption

    Returns:
        Generated RIR of length T as numpy array
        
    Raises:
        ValueError: If parameters are out of valid ranges
    """
    if not (0 < beta < 1):
        raise ValueError("Reflection coefficient beta must be between 0 and 1")
    if T <= 0 or fs <= 0:
        raise ValueError("T and fs must be positive")
    
    c = 343.0  # speed of sound in air (m/s)
    h = np.zeros(T, dtype=np.float32)

    Lx, Ly, Lz = room_dim
    sx, sy, sz = source
    mx, my, mz = mic

    # Validate positions are within room bounds
    if not (0 <= sx <= Lx and 0 <= sy <= Ly and 0 <= sz <= Lz):
        raise ValueError("Source position must be within room bounds")
    if not (0 <= mx <= Lx and 0 <= my <= Ly and 0 <= mz <= Lz):
        raise ValueError("Microphone position must be within room bounds")

    # Loop over image sources
    for nx in range(-reflection_order, reflection_order + 1):
        for ny in range(-reflection_order, reflection_order + 1):
            for nz in range(-reflection_order, reflection_order + 1):

                # Calculate image source position
                img_x = (2 * nx * Lx) + ((-1)**nx) * sx
                img_y = (2 * ny * Ly) + ((-1)**ny) * sy
                img_z = (2 * nz * Lz) + ((-1)**nz) * sz

                # Distance from image source to microphone
                d = np.sqrt((img_x - mx)**2 + (img_y - my)**2 + (img_z - mz)**2)

                # Arrival time in samples
                t_arrival = d / c * fs
                t = int(np.round(t_arrival))

                if 0 <= t < T:
                    # Calculate amplitude with reflection losses and distance attenuation
                    num_reflections = abs(nx) + abs(ny) + abs(nz)
                    reflection_loss = beta ** num_reflections
                    distance_attenuation = 1.0 / (4 * np.pi * d + 1e-9)
                    
                    amplitude = reflection_loss * distance_attenuation
                    h[t] += amplitude

    return h


def generate_dataset(
    num_rooms: int,
    T: int = 512,
    fs: int = 16000,
    room_type: str = "rectangular",
    min_room_size: float = 4.0,
    max_room_size: float = 12.0,
    min_absorption: float = 0.2,
    max_absorption: float = 0.8,
    reflection_order: int = 1,
    seed: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a dataset of RIRs with randomized room configurations.
    
    Args:
        num_rooms: Number of room configurations to generate
        T: Length of each RIR in samples
        fs: Sampling frequency in Hz
        room_type: Type of room geometry (currently only "rectangular" supported)
        min_room_size: Minimum room dimension in meters
        max_room_size: Maximum room dimension in meters
        min_absorption: Minimum absorption coefficient (1-beta)
        max_absorption: Maximum absorption coefficient (1-beta)
        reflection_order: Maximum reflection order for image source method
        seed: Random seed for reproducibility
        
    Returns:
        List of (geometry_features, rir) tuples where:
        - geometry_features: concatenated [room_dims, absorption, source_pos, mic_pos]
        - rir: generated room impulse response
        
    Raises:
        ValueError: If room_type is not supported or parameters are invalid
    """
    if room_type != "rectangular":
        raise ValueError(f"Room type '{room_type}' not supported. Only 'rectangular' is available.")
    
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    
    for i in range(num_rooms):
        # Generate random room dimensions
        room_dim = np.random.uniform(min_room_size, max_room_size, size=3)
        
        # Generate random absorption coefficient
        absorption = np.random.uniform(min_absorption, max_absorption)
        beta = 1.0 - absorption  # Convert absorption to reflection coefficient
        
        # Generate random source and microphone positions within room bounds
        margin = 0.5  # Keep positions away from walls
        source = np.random.uniform(margin, room_dim - margin)
        mic = np.random.uniform(margin, room_dim - margin)
        
        try:
            # Generate RIR using image source method
            h = generate_rir_image(
                T=T,
                fs=fs,
                room_dim=tuple(room_dim),
                source=tuple(source),
                mic=tuple(mic),
                reflection_order=reflection_order,
                beta=beta
            )
            
            # Create feature vector: [room_dims(3), absorption(1), source_pos(3), mic_pos(3)]
            geometry_features = np.concatenate([
                room_dim,
                np.array([absorption]),
                source,
                mic
            ]).astype(np.float32)
            
            data.append((geometry_features, h))
            
        except ValueError as e:
            print(f"Warning: Skipping room {i} due to error: {e}")
            continue
    
    return data
