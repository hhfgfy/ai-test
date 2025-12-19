import numpy as np
labels = np.load("data/labels.npy")  # (N,6) [UP,DOWN,LEFT,RIGHT,ALT,SHIFT]
left_ratio = labels[:,2].mean()
right_ratio = labels[:,3].mean()
print("N:", len(labels))
print("LEFT ratio:", left_ratio, "count:", int(labels[:,2].sum()))
print("RIGHT ratio:", right_ratio, "count:", int(labels[:,3].sum()))
