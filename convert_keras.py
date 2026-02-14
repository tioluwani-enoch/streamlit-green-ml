import tensorflow as tf

print("Loading your model...")
model = tf.keras.models.load_model('waste_sorting_model.keras')

print("Converting to SavedModel format...")
model.export('waste_model_saved')

print("✓ Done! Upload the entire 'waste_model_saved' folder to GitHub")