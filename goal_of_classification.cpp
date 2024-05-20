#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <iomanip>  // for std::fixed and std::setprecision
#include <sstream>  // for std::ostringstream
#include <limits>   // for std::numeric_limits
#include <cmath>    // for std::sqrt

// Define a type for a feature vector, which is a vector of floats
using FeatureVector = std::vector<float>;

// Define a type for the label, which is an integer representing a discrete value
using Label = int;

// Define a type for a data point, which is a struct containing a feature vector and a label
struct DataPoint {
    FeatureVector features;
    Label label;
};

// Define a type for training data, which is a vector of data points
using TrainingData = std::vector<DataPoint>;

// Function to convert a float to a string with fixed precision
std::string float_to_string(float value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value; // Converts float to string with fixed precision
    return out.str(); // Example: 150.0 -> "150.0"
}

// Function to join elements of a vector into a single string with a separator
std::string join_with(const FeatureVector& features, const std::string& separator) {
    std::ostringstream out;
    for (size_t i = 0; i < features.size(); ++i) {
        out << float_to_string(features[i]);
        if (i < features.size() - 1) {
            out << separator; // Add separator except for the last element
        }
    }
    return out.str(); // Example: [150.0, 1.0, 1.0] -> "150.0, 1.0, 1.0"
}

// Function to print the training data for visualization
void print_training_data(const TrainingData& training_data) {
    for (const auto& data_point : training_data) {
        std::cout << "Features: [" << join_with(data_point.features, ", ") << "], Label: " << data_point.label << '\n';
        // Example output: Features: [150.0, 1.0, 1.0], Label: 0
    }
}

// Function to determine if the classification problem is binary
bool is_binary_classification(const TrainingData& training_data) {
    std::set<Label> unique_labels;
    for (const auto& data_point : training_data) {
        unique_labels.insert(data_point.label);
        // Each label is inserted into the set; duplicates are ignored
    }
    return unique_labels.size() == 2; // Binary if there are exactly two unique labels
}

// Function to predict the label for a new data point based on the nearest neighbor
Label predict_label(const FeatureVector& new_features, const TrainingData& training_data) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data is empty"); // Validate that training data is not empty
    }

    if (new_features.empty() || new_features.size() != training_data[0].features.size()) {
        throw std::invalid_argument("Feature vector size mismatch"); // Validate feature vector size
    }

    float min_distance = std::numeric_limits<float>::max();
    Label predicted_label = -1;

    for (const auto& data_point : training_data) {
        float distance = 0.0f;
        for (size_t i = 0; i < new_features.size(); ++i) {
            // Calculate squared Euclidean distance
            distance += (new_features[i] - data_point.features[i]) * (new_features[i] - data_point.features[i]);
            // Example: new_features = {180.0, 1.0, 1.0}, data_point.features = {150.0, 1.0, 1.0}
            // distance = (180.0-150.0)^2 + (1.0-1.0)^2 + (1.0-1.0)^2 = 30.0^2 = 900.0
        }
        distance = std::sqrt(distance); // Taking the square root to get the Euclidean distance
        if (distance < min_distance) {
            min_distance = distance; // Update minimum distance
            predicted_label = data_point.label; // Update predicted label
        }
    }

    return predicted_label; // Return the label of the nearest neighbor
}

int main() {
    // Example training data for fruit classification
    TrainingData fruit_training_data = {
        {{150.0f, 1.0f, 1.0f}, 0}, // Apple
        {{200.0f, 0.0f, 0.0f}, 1}, // Pear
        {{250.0f, 2.0f, 2.0f}, 2}, // Banana
        {{160.0f, 1.0f, 1.0f}, 0}, // Apple
        {{210.0f, 0.0f, 0.0f}, 1}  // Pear
    };

    // Example training data for flower classification
    TrainingData flower_training_data = {
        {{5.1f, 3.5f, 1.4f, 0.2f}, 0}, // Setosa
        {{7.0f, 3.2f, 4.7f, 1.4f}, 1}, // Versicolor
        {{6.3f, 3.3f, 6.0f, 2.5f}, 2}, // Virginica
        {{4.9f, 3.1f, 1.5f, 0.1f}, 0}, // Setosa
        {{6.7f, 3.1f, 4.4f, 1.4f}, 1}  // Versicolor
    };

    std::cout << "Fruit Training Data:\n";
    print_training_data(fruit_training_data); // Prints each fruit's features and label

    if (is_binary_classification(fruit_training_data)) {
        std::cout << "\nThe fruit classification problem is binary.\n";
    } else {
        std::cout << "\nThe fruit classification problem is multi-class.\n";
    }

    std::cout << "\nFlower Training Data:\n";
    print_training_data(flower_training_data); // Prints each flower's features and label

    if (is_binary_classification(flower_training_data)) {
        std::cout << "\nThe flower classification problem is binary.\n";
    } else {
        std::cout << "\nThe flower classification problem is multi-class.\n";
    }

    // Predicting label for a new data point
    try {
        FeatureVector new_fruit = {180.0f, 1.0f, 1.0f}; // Example new fruit features
        Label predicted_label = predict_label(new_fruit, fruit_training_data);
        std::cout << "\nPredicted label for the new fruit: " << predicted_label << '\n'; // Example output: 0 (Apple)
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

    return 0;
}
