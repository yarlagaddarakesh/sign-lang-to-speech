import pickle

# Load the data from the pickle file
with open('data.pickle', 'rb') as f:
    saved_data = pickle.load(f)

# Extract data and labels
data = saved_data['data']
labels = saved_data['labels']

# Check the shape of the data
unique_shapes_count = {}
if data:
    print("Unique Data Shapes:")
    new_data = []
    new_labels = []
    for sample_data, label in zip(data, labels):
        sample_data_shape = len(sample_data)
        if sample_data_shape == 42:
            new_data.append(sample_data)
            new_labels.append(label)
        if sample_data_shape not in unique_shapes_count:
            unique_shapes_count[sample_data_shape] = 0
        unique_shapes_count[sample_data_shape] += 1
    
    for shape, count in unique_shapes_count.items():
        print(f"- Shape {shape}: {count} samples")

    # Save the extracted data with shape 42 and corresponding labels into a new pickle file
    with open('data_shape_42.pickle', 'wb') as new_f:
        pickle.dump({'data': new_data, 'labels': new_labels}, new_f)
    print("Extracted data with shape 42 and corresponding labels saved in 'data_shape_42.pickle'")
else:
    print("No data found in the pickle file.")

# Check the shape of the labels
if labels:
    print(f"Number of labels: {len(labels)}")
else:
    print("No labels found in the pickle file.")
