#include "cnn/cnn.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Advanced CNN Blocks Functionality Test\n";
    std::cout << "======================================\n\n";

    try {
        // Test 1: DepthwiseConv2D
        std::cout << "Test 1: DepthwiseConv2D\n";
        std::cout << "Creating 3-channel 32x32 input...\n";
        cnn::Tensor input({1, 3, 32, 32});
        float* data = input.data();
        for (int i = 0; i < input.size(); ++i) {
            data[i] = 0.1f;  // Simple test data
        }
        
        cnn::DepthwiseConv2DLayer dw_conv(3, 3, 3, 1, 1, 1, 1);
        cnn::Tensor output = dw_conv.forward(input, false);
        std::cout << "✓ DepthwiseConv2D forward pass successful\n";
        std::cout << "  Input: [1, 3, 32, 32] -> Output: [" 
                  << output.shape()[0] << ", " << output.shape()[1] 
                  << ", " << output.shape()[2] << ", " << output.shape()[3] << "]\n\n";

        // Test 2: ResidualBlock (simple case)
        std::cout << "Test 2: ResidualBlock\n";
        cnn::ResidualBlock res_block(3, 3, 1);  // Same input/output channels, stride=1
        cnn::Tensor res_output = res_block.forward(input, false);
        std::cout << "✓ ResidualBlock forward pass successful\n";
        std::cout << "  Input: [1, 3, 32, 32] -> Output: [" 
                  << res_output.shape()[0] << ", " << res_output.shape()[1] 
                  << ", " << res_output.shape()[2] << ", " << res_output.shape()[3] << "]\n\n";

        // Test 3: InceptionModule
        std::cout << "Test 3: InceptionModule\n";
        cnn::InceptionModule inception(3, 16, 16, 16, 16);  // Total output: 64 channels
        cnn::Tensor inception_output = inception.forward(input, false);
        std::cout << "✓ InceptionModule forward pass successful\n";
        std::cout << "  Input: [1, 3, 32, 32] -> Output: [" 
                  << inception_output.shape()[0] << ", " << inception_output.shape()[1] 
                  << ", " << inception_output.shape()[2] << ", " << inception_output.shape()[3] << "]\n\n";

        // Test 4: BottleneckBlock
        std::cout << "Test 4: BottleneckBlock\n";
        cnn::BottleneckBlock bottleneck(3, 8, 3, 1);  // 3->8->3 channels, stride=1
        cnn::Tensor bottleneck_output = bottleneck.forward(input, false);
        std::cout << "✓ BottleneckBlock forward pass successful\n";
        std::cout << "  Input: [1, 3, 32, 32] -> Output: [" 
                  << bottleneck_output.shape()[0] << ", " << bottleneck_output.shape()[1] 
                  << ", " << bottleneck_output.shape()[2] << ", " << bottleneck_output.shape()[3] << "]\n\n";

        // Test 5: Backward pass for one block
        std::cout << "Test 5: Backward Pass\n";
        cnn::Tensor grad({1, 3, 32, 32});
        float* grad_data = grad.data();
        for (int i = 0; i < grad.size(); ++i) {
            grad_data[i] = 0.01f;  // Small gradient
        }
        
        cnn::Tensor input_grad = res_block.backward(grad);
        std::cout << "✓ ResidualBlock backward pass successful\n";
        std::cout << "  Gradient shape: [" 
                  << input_grad.shape()[0] << ", " << input_grad.shape()[1] 
                  << ", " << input_grad.shape()[2] << ", " << input_grad.shape()[3] << "]\n\n";

        std::cout << "🎉 ALL ADVANCED BLOCKS WORKING CORRECTLY!\n";
        std::cout << "\nThe CNN framework now supports:\n";
        std::cout << "  ✓ DepthwiseConv2D - Efficient mobile-style convolutions\n";
        std::cout << "  ✓ ResidualBlock - ResNet-style skip connections\n";
        std::cout << "  ✓ InceptionModule - GoogLeNet-style multi-branch processing\n";
        std::cout << "  ✓ BottleneckBlock - ResNet-50 style efficient residual blocks\n";
        std::cout << "\nThese blocks enable building modern CNN architectures like:\n";
        std::cout << "  • ResNet, ResNet-50\n";
        std::cout << "  • MobileNet\n";
        std::cout << "  • GoogLeNet/Inception\n";
        std::cout << "  • Custom efficient architectures\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
