import numpy as np

def generate_rir(T=50, num_echoes=2, decay=0.9, seed=None):
    if seed is not None:
        np.random.seed(seed)
    h = np.zeros(T)
    delays = np.random.randint(0, T, size=num_echoes)
    amplitudes = decay ** np.arange(num_echoes) * np.random.rand(num_echoes)
    for d, a in zip(delays, amplitudes):
        if d < T:
            h[d] += a
    return h

def generate_dataset(num_rooms, T=512, room_type='rectangular'):
    data = []
    for _ in range(num_rooms):
        gamma = np.random.rand(3) * 10
        alpha = np.random.rand(1)
        s = np.random.rand(3) * 5
        m = np.random.rand(3) * 5
        h = generate_rir(T=T)
        x = np.concatenate([gamma, alpha, s, m])
        data.append((x.astype(np.float32), h.astype(np.float32)))
    return data
