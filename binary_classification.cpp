#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <iomanip>  // for std::fixed and std::setprecision
#include <sstream>  // for std::ostringstream

// Define a type for a feature vector, a vector of floats
using FeatureVector = std::vector<float>;

// Define a type for the label, an integer representing a discrete value
using Label = int;

// Define a type for a data point, a struct of feature vector and label
struct DataPoint {
    FeatureVector features;
    Label label;
};

// Define a type for training data, a vector of data points
using TrainingData = std::vector<DataPoint>;

// Function to convert a float to a string with fixed precision
std::string float_to_string(float value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

// Function to join elements of a vector into a single string with a separator
std::string join_with(const FeatureVector& features, const std::string& separator) {
    std::ostringstream out;
    for (size_t i = 0; i < features.size(); ++i) {
        out << float_to_string(features[i]);
        if (i < features.size() - 1) {
            out << separator;
        }
    }
    return out.str();
}

// Function to print the training data for visualization
void print_training_data(const TrainingData& training_data) {
    for (const auto& data_point : training_data) {
        std::cout << "Features: [" << join_with(data_point.features, ", ") << "], Label: " << data_point.label << '\n';
    }
}

// Function to determine if the classification problem is binary
bool is_binary_classification(const TrainingData& training_data) {
    std::set<Label> unique_labels;
    for (const auto& data_point : training_data) {
        unique_labels.insert(data_point.label);
    }
    return unique_labels.size() == 2;
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
    print_training_data(fruit_training_data);

    if (is_binary_classification(fruit_training_data)) {
        std::cout << "\nThe fruit classification problem is binary.\n";
    } else {
        std::cout << "\nThe fruit classification problem is multi-class.\n";
    }

    std::cout << "\nFlower Training Data:\n";
    print_training_data(flower_training_data);

    if (is_binary_classification(flower_training_data)) {
        std::cout << "\nThe flower classification problem is binary.\n";
    } else {
        std::cout << "\nThe flower classification problem is multi-class.\n";
    }

    return 0;
}
