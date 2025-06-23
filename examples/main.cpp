#include <iostream>
#include <vector>
#include <random>
#include "cnn/cnn.h"

using namespace cnn;

// Generate synthetic data for testing
std::pair<Tensor, Tensor> generate_dummy_data(int batch_size, int input_size, int num_classes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Generate random input data
    Tensor inputs({batch_size, input_size});
    for (int i = 0; i < inputs.size(); ++i) {
        inputs.data()[i] = dis(gen);
    }
    
    // Generate random one-hot targets
    Tensor targets({batch_size, num_classes});
    targets.zero();
    
    std::uniform_int_distribution<int> class_dis(0, num_classes - 1);
    for (int i = 0; i < batch_size; ++i) {
        int class_idx = class_dis(gen);
        targets.at(i, class_idx) = 1.0f;
    }
    
    return {inputs, targets};
}

int main() {
    std::cout << "CNN Framework Example" << std::endl;
    std::cout << "=====================" << std::endl;
    
    try {
        // Create a simple neural network for classification
        Model model;
        
        // Input: 784 features (e.g., 28x28 image flattened)
        // Hidden layer: 128 neurons with ReLU
        // Output: 10 classes
        model.add_layer<LinearLayer>(784, 128);
        model.add_layer<ReLULayer>();
        model.add_layer<DropoutLayer>(0.2f);  // 20% dropout
        model.add_layer<LinearLayer>(128, 64);
        model.add_layer<ReLULayer>();
        model.add_layer<LinearLayer>(64, 10);
        model.add_layer<SoftmaxLayer>();
        
        std::cout << "\nModel architecture:" << std::endl;
        model.summary();
        
        // Create optimizer and loss function
        auto optimizer = make_adam_optimizer(
            model.get_parameters(),
            model.get_gradients(),
            0.001f  // learning rate
        );
        
        auto loss_fn = make_crossentropy_loss();
        
        // Training parameters
        const int batch_size = 32;
        const int epochs = 5;
        const int batches_per_epoch = 100;
        
        std::cout << "\nStarting training..." << std::endl;
        
        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            for (int batch = 0; batch < batches_per_epoch; ++batch) {
                // Generate dummy data
                auto [inputs, targets] = generate_dummy_data(batch_size, 784, 10);
                
                // Forward pass
                model.set_training(true);
                auto predictions = model.forward(inputs);
                
                // Compute loss
                float loss = loss_fn->compute(predictions, targets);
                epoch_loss += loss;
                
                // Backward pass
                optimizer->zero_grad();
                auto loss_grad = loss_fn->gradient(predictions, targets);
                model.backward(loss_grad);
                
                // Update weights
                optimizer->step();
                
                if (batch % 20 == 0) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << ", Batch " << batch + 1 << "/" << batches_per_epoch
                              << ", Loss: " << loss << std::endl;
                }
            }
            
            float avg_loss = epoch_loss / batches_per_epoch;
            std::cout << "Epoch " << epoch + 1 << " completed. Average loss: " << avg_loss << std::endl;
        }
        
        std::cout << "\nTraining completed!" << std::endl;
        
        // Test inference
        std::cout << "\nTesting inference..." << std::endl;
        model.set_training(false);
        
        auto [test_inputs, test_targets] = generate_dummy_data(5, 784, 10);
        auto test_predictions = model.forward(test_inputs);
        
        std::cout << "Test predictions shape: [" << test_predictions.shape()[0] 
                  << ", " << test_predictions.shape()[1] << "]" << std::endl;
        
        // Calculate accuracy (simplified)
        int correct = 0;
        for (int i = 0; i < 5; ++i) {
            // Find predicted class
            int pred_class = 0;
            float max_pred = test_predictions.at(i, 0);
            for (int j = 1; j < 10; ++j) {
                if (test_predictions.at(i, j) > max_pred) {
                    max_pred = test_predictions.at(i, j);
                    pred_class = j;
                }
            }
            
            // Find true class
            int true_class = 0;
            for (int j = 0; j < 10; ++j) {
                if (test_targets.at(i, j) > 0.5f) {
                    true_class = j;
                    break;
                }
            }
            
            if (pred_class == true_class) correct++;
            
            std::cout << "Sample " << i << ": Predicted " << pred_class 
                      << ", True " << true_class << std::endl;
        }
        
        std::cout << "Test accuracy: " << (correct * 100.0f / 5) << "%" << std::endl;
        
        // Save model
        std::cout << "\nSaving model..." << std::endl;
        model.save("trained_model.bin");
        
        std::cout << "\nExample completed successfully! 🎉" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
