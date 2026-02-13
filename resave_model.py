import tensorflow as tf

# Load your current model
print("Loading model...")
model = tf.keras.models.load_model('waste_sorting_model.keras')

# Save in .h5 format (most compatible)
print("Re-saving model...")
model.save('waste_sorting_model_compatible.h5')

print("✓ Done! Upload waste_sorting_model_compatible.h5 to GitHub")