# Cell 10: Evaluate Both Watermarks

print("--- Calculating Evaluation Metrics ---")

# --- Grayscale Watermark Evaluation ---
print("\n--- Grayscale Watermark ---")
original_gs_flat = grayscale_data[:len(extracted_pixels_grayscale)]
extracted_gs_flat = extracted_pixels_grayscale.flatten()

# 1. RMSE
rmse_gs = np.sqrt(np.mean((original_gs_flat - extracted_gs_flat) ** 2))
print(f" - RMSE: {rmse_gs:.4f}")

# 2. NCC
ncc_gs = np.corrcoef(original_gs_flat, extracted_gs_flat)[0, 1]
print(f" - NCC: {ncc_gs:.4f}")

# 3. PSNR
psnr_gs = calculate_psnr(np.array(grayscale_img), extracted_grayscale_img_array)
print(f" - PSNR: {psnr_gs:.4f} dB")


# --- Color Watermark Evaluation ---
print("\n--- Color Watermark ---")
original_color_flat = color_data[:len(extracted_pixels_color)]
extracted_color_flat = extracted_pixels_color.flatten()

# 1. RMSE
rmse_color = np.sqrt(np.mean((original_color_flat - extracted_color_flat) ** 2))
print(f" - RMSE: {rmse_color:.4f}")

# 2. NCC
ncc_color = np.corrcoef(original_color_flat, extracted_color_flat)[0, 1]
print(f" - NCC: {ncc_color:.4f}")

# 3. PSNR
psnr_color = calculate_psnr(np.array(color_img), extracted_color_img_array)
print(f" - PSNR: {psnr_color:.4f} dB")


# --- Overall Model Metrics ---
print("\n--- Overall Model Metrics ---")
# 4. Hausdorff Distance (compares original model to the final watermarked model)
# This assumes 'watermarked_vertices' is from the non-attacked, reloaded mesh
hausdorff_dist_1 = directed_hausdorff(original_vertices, watermarked_vertices)[0]
hausdorff_dist_2 = directed_hausdorff(watermarked_vertices, original_vertices)[0]
hausdorff_distance = max(hausdorff_dist_1, hausdorff_dist_2)
print(f" - Hausdorff Distance: {hausdorff_distance:.6f}")

# 5. Bit Embedding Rate (total bits embedded from both watermarks)
total_bits_embedded = (len(target_indices_color) + len(target_indices_grayscale)) * 8
bit_embedding_rate = total_bits_embedded / len(vertices)
print(f" - Bit Embedding Rate: {bit_embedding_rate:.4f} bits per vertex")
