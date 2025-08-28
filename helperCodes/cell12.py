# Cell 12: Extract Both Watermarks After Attack

# --- Specify which attacked model to load ---
# This MUST match the 'output_filename' from your attack cell (Cell 11)
filename_to_evaluate = 'rotated_90_model.obj'
# --------------------------------------------

print(f"--- Extracting Both Watermarks from Attacked Model: '{filename_to_evaluate}' ---")

# 1. Load the specified attacked model
try:
    loaded_attacked_mesh = trimesh.load_mesh(filename_to_evaluate)
    print(f"Successfully loaded '{filename_to_evaluate}'.")
    display(loaded_attacked_mesh.scene().show())
except Exception as e:
    print(f"ERROR: Could not load '{filename_to_evaluate}'. Please run the attack cell first.")
    # Stop execution if the file doesn't exist
    raise e

# --- Setup for Extraction ---
from scipy.spatial import KDTree
original_vertices = mesh.vertices
attacked_vertices = loaded_attacked_mesh.vertices.astype(np.float64)
attacked_tree = KDTree(attacked_vertices)
attacked_normals = loaded_attacked_mesh.vertex_normals

# --- Helper Function to Extract a Single Watermark ---
def extract_watermark(target_indices):
    extracted_data = []
    for i, vertex_idx in enumerate(target_indices):
        original_v = original_vertices[vertex_idx]
        distance, attacked_idx = attacked_tree.query(original_v)
        attacked_v = attacked_vertices[attacked_idx]
        normal_v = attacked_normals[attacked_idx]
        
        displacement_vector = attacked_v - original_v
        projected_displacement = np.dot(displacement_vector, normal_v)
        extracted_value = projected_displacement / embedding_strength
        extracted_data.append(extracted_value)
    
    normalized_data = np.array(extracted_data).reshape(-1, 1)
    pixel_data = scaler.inverse_transform(normalized_data)
    return pixel_data

# --- Extract Both Watermarks ---
extracted_pixels_color = extract_watermark(target_indices_color)
extracted_pixels_grayscale = extract_watermark(target_indices_grayscale)
print("Extraction complete.")

# --- Reshape and Visualize ---
extracted_color_array = extracted_pixels_color.flatten()
extracted_color_img_array = np.clip(extracted_color_array, 0, 255).astype('uint8').reshape((*IMAGE_DIMENSION, 3))

extracted_grayscale_array = extracted_pixels_grayscale.flatten()
extracted_grayscale_img_array = np.clip(extracted_grayscale_array, 0, 255).astype('uint8').reshape(IMAGE_DIMENSION)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(extracted_grayscale_img_array, cmap='gray')
ax[0].set_title("Extracted Grayscale (Attacked)")
ax[0].axis('off')
ax[1].imshow(extracted_color_img_array)
ax[1].set_title("Extracted Color (Attacked)")
ax[1].axis('off')
plt.show()

print("\nPost-attack extraction finished. Re-run Cell 10 to evaluate robustness.")
