#include "cnn/cnn.h"
#include <iostream>

using namespace cnn;

int main() {
    std::cout << "Advanced CNN Blocks Compilation Test" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        // This test just verifies that the advanced block classes can be instantiated
        // and that the basic interface works without running full training
        
        std::cout << "Testing DepthwiseConv2DLayer instantiation..." << std::endl;
        {
            Model model;
            model.add_layer<DepthwiseConv2DLayer>(3, 3, 3, 1, 1, 1, 1);
            std::cout << "✓ DepthwiseConv2DLayer created successfully" << std::endl;
        }
        
        std::cout << "Testing ResidualBlock instantiation..." << std::endl;
        {
            Model model;
            model.add_layer<ResidualBlock>(16, 16, 1);
            std::cout << "✓ ResidualBlock created successfully" << std::endl;
        }
        
        std::cout << "Testing InceptionModule instantiation..." << std::endl;
        {
            Model model;
            model.add_layer<InceptionModule>(64, 16, 32, 8, 16);
            std::cout << "✓ InceptionModule created successfully" << std::endl;
        }
        
        std::cout << "Testing BottleneckBlock instantiation..." << std::endl;
        {
            Model model;
            model.add_layer<BottleneckBlock>(64, 16, 64, 1);
            std::cout << "✓ BottleneckBlock created successfully" << std::endl;
        }
        
        std::cout << "\n🎉 All advanced blocks compiled successfully!" << std::endl;
        std::cout << "The CNN framework now includes:" << std::endl;
        std::cout << "  • DepthwiseConv2D (for efficient mobile architectures)" << std::endl;
        std::cout << "  • ResidualBlock (ResNet-style skip connections)" << std::endl;
        std::cout << "  • InceptionModule (GoogLeNet-style multi-branch convolution)" << std::endl;
        std::cout << "  • BottleneckBlock (ResNet-50 style efficient residual blocks)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
