from datetime import datetime

# Get the current date and time in your desired format
current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

# Use the formatted datetime string in your callback filepath
weights_dir = f"weights/model_{current_datetime}.h5"
train_dir = 'src/train/'
val_dir = 'src/test/'
logs_dir='logs/'