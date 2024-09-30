import random
import numpy as np


# ==========================================================
#  Generate balanced Siamese pairs for multiclass
# ==========================================================
def generate_balanced_siamese_pairs(data, labels, num_pairs, num_classes):
    # Initialize pairs and labels
    pairs = []
    pair_labels = []

    # Set to keep track of generated pairs
    generated_pairs = set()

    # Create a dictionary to hold indices for each class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Calculate number of positive and negative pairs
    num_positive_pairs = num_pairs // 2
    num_negative_pairs = num_pairs - num_positive_pairs

    # Function to create a unique key for a pair
    def make_pair_key(idx1, idx2):
        return tuple(sorted((int(idx1), int(idx2))))

    # Maximum number of failed attempts to prevent infinite loops
    max_attempts = num_pairs * 10  # Adjust as needed

    # Generate positive pairs
    positive_pairs_generated = 0
    attempts = 0  # Reset attempts for positive pairs
    while positive_pairs_generated < num_positive_pairs and attempts < max_attempts:
        # Select a random class
        class_label = random.choice(range(num_classes))
        # Check if there are at least 2 samples in this class
        if len(class_indices[class_label]) < 2:
            attempts += 1
            continue
        idx1, idx2 = np.random.choice(class_indices[class_label], size=2, replace=False)
        pair_key = make_pair_key(idx1, idx2)
        if pair_key in generated_pairs:
            attempts += 1
            continue
        # Add the pair and label
        pairs.append([data[idx1], data[idx2]])
        pair_labels.append(1)  # Same class
        generated_pairs.add(pair_key)
        positive_pairs_generated += 1
        attempts = 0  # Reset attempts since we successfully added a pair

    if attempts >= max_attempts:
        print("Reached maximum attempts while generating positive pairs.")

    # Generate negative pairs
    negative_pairs_generated = 0
    attempts = 0  # Reset attempts for negative pairs
    while negative_pairs_generated < num_negative_pairs and attempts < max_attempts:
        # Select two different classes
        class_label1, class_label2 = random.sample(range(num_classes), 2)
        if len(class_indices[class_label1]) == 0 or len(class_indices[class_label2]) == 0:
            attempts += 1
            continue
        idx1 = np.random.choice(class_indices[class_label1])
        idx2 = np.random.choice(class_indices[class_label2])
        pair_key = make_pair_key(idx1, idx2)
        if pair_key in generated_pairs:
            attempts += 1
            continue
        # Add the pair and label
        pairs.append([data[idx1], data[idx2]])
        pair_labels.append(0)  # Different classes
        generated_pairs.add(pair_key)
        negative_pairs_generated += 1
        attempts = 0  # Reset attempts since we successfully added a pair

    if attempts >= max_attempts:
        print("Reached maximum attempts while generating negative pairs.")

    if positive_pairs_generated < num_positive_pairs or negative_pairs_generated < num_negative_pairs:
        print("Could not generate all unique pairs requested.")

    return np.array(pairs), np.array(pair_labels)

