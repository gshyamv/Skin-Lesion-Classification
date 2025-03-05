import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

# Generate imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, 
                           weights=[0.1, 0.9], n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, 
                           n_samples=200, random_state=42)

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Identify original and synthetic points
num_original = len(X)
synthetic_points = X_resampled[num_original:]

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_title("SMOTE Animation")

# Plot original data
scatter_real = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={0: "red", 1: "blue"}, alpha=0.7, ax=ax)
scatter_synthetic, = ax.plot([], [], "go", alpha=0.5, markersize=8, label="Synthetic Data")

# Animation function
frames = []
def update(frame):
    if frame == 0:
        return scatter_synthetic,
    scatter_synthetic.set_data(synthetic_points[:frame, 0], synthetic_points[:frame, 1])
    
    # Capture the frame
    fig.canvas.draw()
    frame_img = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to numpy array
    frames.append(frame_img)  # Store frames for saving as a video
    
    return scatter_synthetic,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(synthetic_points) + 1, interval=100, repeat=False)

# Tkinter Window
root = tk.Tk()
root.title("SMOTE Animation")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def save_video():
    """Save frames as video using OpenCV."""
    if not frames:
        print("No frames captured!")
        return
    
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter("smote_animation.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR for OpenCV
        out.write(frame_bgr)
    
    out.release()
    print("Video saved as smote_animation.avi")

# Save Button
btn_save = tk.Button(root, text="Save Video", command=save_video)
btn_save.pack()

root.mainloop()
