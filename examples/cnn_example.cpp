#include <iostream>
#include <vector>
#include <random>
#include "cnn/cnn.h"

using namespace cnn;

// Generate synthetic CIFAR-10 like data
std::pair<Tensor, Tensor> generate_cifar_like_data(int batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Generate 32x32x3 RGB images
    Tensor inputs({batch_size, 3, 32, 32});
    for (int i = 0; i < inputs.size(); ++i) {
        inputs.data()[i] = dis(gen);
    }
    
    // Generate one-hot labels for 10 classes
    Tensor targets({batch_size, 10});
    targets.zero();
    
    std::uniform_int_distribution<int> class_dis(0, 9);
    for (int i = 0; i < batch_size; ++i) {
        int class_idx = class_dis(gen);
        targets.at(i, class_idx) = 1.0f;
    }
    
    return {inputs, targets};
}

int main() {
    std::cout << "CNN Convolutional Example" << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        // Create a CNN similar to a simple LeNet/AlexNet architecture
        Model cnn_model;
        
        // First convolutional block
        cnn_model.add_layer<Conv2DLayer>(3, 16, 5, 5, 1, 1, 2, 2);  // 3->16 channels, 5x5 kernel, stride 1, pad 2
        cnn_model.add_layer<ReLULayer>();
        cnn_model.add_layer<MaxPoolingLayer>(2);  // 2x2 max pooling
        
        // Second convolutional block
        cnn_model.add_layer<Conv2DLayer>(16, 32, 5, 5, 1, 1, 2, 2); // 16->32 channels
        cnn_model.add_layer<ReLULayer>();
        cnn_model.add_layer<MaxPoolingLayer>(2);
        
        // Third convolutional block
        cnn_model.add_layer<Conv2DLayer>(32, 64, 3, 3, 1, 1, 1, 1); // 32->64 channels, 3x3 kernel
        cnn_model.add_layer<ReLULayer>();
        
        // Global average pooling (simulate with avg pooling)
        cnn_model.add_layer<AvgPoolingLayer>(8);  // 8x8 average pooling
        
        // Flatten and classify
        cnn_model.add_layer<FlattenLayer>();
        cnn_model.add_layer<LinearLayer>(64, 128);
        cnn_model.add_layer<ReLULayer>();
        cnn_model.add_layer<DropoutLayer>(0.5f);
        cnn_model.add_layer<LinearLayer>(128, 10);
        cnn_model.add_layer<SoftmaxLayer>();
        
        std::cout << "\nCNN Model Architecture:" << std::endl;
        cnn_model.summary();
        
        // Create optimizer with lower learning rate for CNN
        auto optimizer = make_adam_optimizer(
            cnn_model.get_parameters(),
            cnn_model.get_gradients(),
            0.0001f  // Lower learning rate for CNN
        );
        
        auto loss_fn = make_crossentropy_loss();
        
        // Training parameters
        const int batch_size = 8;  // Smaller batch size due to memory
        const int epochs = 3;
        const int batches_per_epoch = 20;
        
        std::cout << "\nStarting CNN training..." << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Input shape: [" << batch_size << ", 3, 32, 32]" << std::endl;
        
        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            for (int batch = 0; batch < batches_per_epoch; ++batch) {
                // Generate synthetic CIFAR-10 like data
                auto [inputs, targets] = generate_cifar_like_data(batch_size);
                
                // Forward pass
                cnn_model.set_training(true);
                auto predictions = cnn_model.forward(inputs);
                
                // Compute loss
                float loss = loss_fn->compute(predictions, targets);
                epoch_loss += loss;
                
                // Backward pass
                optimizer->zero_grad();
                auto loss_grad = loss_fn->gradient(predictions, targets);
                cnn_model.backward(loss_grad);
                
                // Update weights
                optimizer->step();
                
                if (batch % 5 == 0) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << ", Batch " << batch + 1 << "/" << batches_per_epoch
                              << ", Loss: " << loss << std::endl;
                }
            }
            
            float avg_loss = epoch_loss / batches_per_epoch;
            std::cout << "Epoch " << epoch + 1 << " completed. Average loss: " << avg_loss << std::endl;
        }
        
        std::cout << "\nCNN Training completed!" << std::endl;
        
        // Test inference with different data
        std::cout << "\nTesting CNN inference..." << std::endl;
        cnn_model.set_training(false);
        
        auto [test_inputs, test_targets] = generate_cifar_like_data(3);
        auto test_predictions = cnn_model.forward(test_inputs);
        
        std::cout << "Test predictions shape: [" << test_predictions.shape()[0] 
                  << ", " << test_predictions.shape()[1] << "]" << std::endl;
        
        // Show some prediction results
        for (int i = 0; i < 3; ++i) {
            int pred_class = 0;
            float max_pred = test_predictions.at(i, 0);
            for (int j = 1; j < 10; ++j) {
                if (test_predictions.at(i, j) > max_pred) {
                    max_pred = test_predictions.at(i, j);
                    pred_class = j;
                }
            }
            
            int true_class = 0;
            for (int j = 0; j < 10; ++j) {
                if (test_targets.at(i, j) > 0.5f) {
                    true_class = j;
                    break;
                }
            }
            
            std::cout << "Sample " << i << ": Predicted class " << pred_class 
                      << " (confidence: " << max_pred << "), True class " << true_class << std::endl;
        }
        
        // Save the CNN model
        std::cout << "\nSaving CNN model..." << std::endl;
        cnn_model.save("cnn_model.bin");
        
        std::cout << "\nCNN Example completed successfully! 🎉" << std::endl;
        std::cout << "\nThis demonstrates:" << std::endl;
        std::cout << "- Convolutional layers with im2col optimization" << std::endl;
        std::cout << "- Multiple pooling operations" << std::endl;
        std::cout << "- Proper gradient flow through CNN architecture" << std::endl;
        std::cout << "- Memory-efficient 4D tensor operations" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
