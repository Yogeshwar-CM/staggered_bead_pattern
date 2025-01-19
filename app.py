import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, render_template, send_file

# Fix Matplotlib GUI issue
import matplotlib
matplotlib.use("Agg")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clear_uploads():
    """Delete all files in the uploads folder before processing a new image."""
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)  # Delete folder and contents
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Recreate empty folder

def load_image(image_path, grid_width, grid_height):
    """Load and resize the image to fit the bead grid."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((grid_width, grid_height), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None

def quantize_colors(img_array, palette):
    """Optimized: Map image colors to the closest available bead colors using NumPy."""
    img_flat = img_array.reshape(-1, 3)  # Flatten for batch processing
    palette = np.array(palette)

    # Compute distances efficiently
    distances = np.linalg.norm(img_flat[:, None] - palette[None, :], axis=2)
    closest_colors = palette[np.argmin(distances, axis=1)]

    return closest_colors.reshape(img_array.shape)

def draw_staggered_grid(img_array, bead_size, gap=2):
    """Draw the bead pattern with staggered rows (peyote brick pattern), visible grid, and scale."""
    grid_height, grid_width, _ = img_array.shape
    fig, ax = plt.subplots(figsize=(12, 12))  # Increased size for better scale visibility
    ax.set_aspect('equal')

    # Set up grid
    ax.set_xticks(np.arange(0, grid_width * bead_size, bead_size))
    ax.set_yticks(np.arange(0, grid_height * bead_size, bead_size))
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.set_axisbelow(True)

    # Draw beads with staggered (peyote) layout
    for y in range(grid_height):
        for x in range(grid_width):
            color = img_array[y, x] / 255
            x_pos = x * bead_size + (bead_size / 2 if y % 2 == 1 else 0)  # Stagger every other row
            y_pos = (grid_height - 1 - y) * bead_size  # Reverse the y-axis to fix inversion
            bead_radius = (bead_size - gap) / 2
            circle = plt.Circle((x_pos, y_pos), bead_radius, color=color, edgecolor="black", linewidth=0.5)
            ax.add_artist(circle)

    # Set image boundaries and turn off axis
    plt.xlim(-bead_size, grid_width * bead_size)
    plt.ylim(-bead_size, grid_height * bead_size)
    plt.axis('off')

    return fig

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        clear_uploads()  # Clear the uploads folder before processing new image

        image_file = request.files.get("image")
        grid_width = request.form.get("grid_width", type=int)
        grid_height = request.form.get("grid_height", type=int)

        if not image_file:
            return render_template("index.html", error="Please upload an image.")

        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        img_array = load_image(image_path, grid_width, grid_height)
        if img_array is None:
            return render_template("index.html", error="Error loading image.")

        # Define a simple color palette for beads
        palette = [
            [255, 0, 0],   # Red
            [0, 255, 0],   # Green
            [0, 0, 255],   # Blue
            [255, 255, 0], # Yellow
            [255, 165, 0], # Orange
            [128, 0, 128], # Purple
            [255, 192, 203] # Pink
        ]

        img_quantized = quantize_colors(img_array, palette)
        bead_size = 10  # Fixed bead size of 10
        fig = draw_staggered_grid(img_quantized, bead_size)

        output_path = os.path.join(UPLOAD_FOLDER, "bead_pattern.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)

        # Count beads per color
        bead_counts = {}
        for row in img_quantized:
            for color in row:
                color_tuple = tuple(color)
                bead_counts[color_tuple] = bead_counts.get(color_tuple, 0) + 1

        # Prepare color list with counts
        color_list = sorted(bead_counts.items(), key=lambda x: x[1], reverse=True)

        return render_template("index.html", image_url="uploads/bead_pattern.png", color_list=color_list)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype="image/png")

def handler(event, context):
    with app.test_request_context(event['path'], method=event['httpMethod']):
        return app.full_dispatch_request()

if __name__ == "__main__":
    app.run(debug=True)
