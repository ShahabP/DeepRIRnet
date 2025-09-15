import numpy as np

def generate_rir_image(T=512, fs=16000, room_dim=(10, 8, 3), source=(2, 3, 1.5), mic=(5, 4, 1.5), reflection_order=1, beta=0.7):
    """
    Generate a Room Impulse Response (RIR) using a simplified image-source model.

    Args:
        T (int): Length of the impulse response.
        fs (int): Sampling frequency.
        room_dim (tuple): Room dimensions (Lx, Ly, Lz).
        source (tuple): Source position (x, y, z).
        mic (tuple): Microphone position (x, y, z).
        reflection_order (int): Maximum order of reflections.
        beta (float): Reflection coefficient (0 < beta < 1).

    Returns:
        h (np.ndarray): Generated RIR of length T.
    """
    c = 343.0  # speed of sound (m/s)
    h = np.zeros(T)

    Lx, Ly, Lz = room_dim
    sx, sy, sz = source
    mx, my, mz = mic

    # Loop over image sources
    for nx in range(-reflection_order, reflection_order + 1):
        for ny in range(-reflection_order, reflection_order + 1):
            for nz in range(-reflection_order, reflection_order + 1):

                # Image source position
                img_x = (2 * nx * Lx) + ((-1)**nx) * sx
                img_y = (2 * ny * Ly) + ((-1)**ny) * sy
                img_z = (2 * nz * Lz) + ((-1)**nz) * sz

                # Distance to microphone
                d = np.sqrt((img_x - mx)**2 + (img_y - my)**2 + (img_z - mz)**2)

                # Arrival time (samples)
                t = int(np.round(d / c * fs))

                if t < T:
                    # Attenuation: distance + reflection loss
                    refl = beta ** (abs(nx) + abs(ny) + abs(nz))
                    amp = refl / (4 * np.pi * d + 1e-9)
                    h[t] += amp

    return h.astype(np.float32)


def generate_dataset(num_rooms, T=512, fs=16000, room_type='rectangular'):
    data = []
    for _ in range(num_rooms):
        # Randomize room, source, mic
        room_dim = np.random.uniform(4, 12, size=3)  # room size between 4–12 m
        source = np.random.uniform(0.5, room_dim - 0.5)
        mic = np.random.uniform(0.5, room_dim - 0.5)

        # Generate RIR with image model
        h = generate_rir_image(T=T, fs=fs, room_dim=tuple(room_dim),
                               source=tuple(source), mic=tuple(mic),
                               reflection_order=1, beta=0.8)

        # Feature vector (room + source + mic)
        x = np.concatenate([room_dim, source, mic])
        data.append((x.astype(np.float32), h))

    return data
