#include "cnn/cnn.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace cnn;

void run_layer_tests() {
    std::cout << "Running Layer tests..." << std::endl;
    
    // Test Linear layer
    {
        LinearLayer linear(10, 5);
        Tensor input({2, 10});  // batch size 2
        input.fill(1.0f);
        
        Tensor output = linear.forward(input, true);
        assert(output.shape()[0] == 2);
        assert(output.shape()[1] == 5);
        
        std::cout << "  Linear layer test passed" << std::endl;
    }
    
    // Test Conv2D layer
    {
        Conv2DLayer conv(3, 16, 3, 3, 1, 1, 1, 1);
        Tensor input({1, 3, 32, 32});
        input.fill(1.0f);
        
        Tensor output = conv.forward(input, true);
        assert(output.shape()[0] == 1);
        assert(output.shape()[1] == 16);
        assert(output.shape()[2] == 32);
        assert(output.shape()[3] == 32);
        
        std::cout << "  Conv2D layer test passed" << std::endl;
    }
    
    // Test activation functions
    {
        ReLULayer relu;
        Tensor input({2, 5});
        input.fill(-1.0f);
        input.at(0, 0) = 2.0f;
        
        Tensor output = relu.forward(input, true);
        assert(output.at(0, 0) == 2.0f);  // Positive value unchanged
        assert(output.at(0, 1) == 0.0f);  // Negative value clipped
        
        std::cout << "  ReLU activation test passed" << std::endl;
    }
    
    std::cout << "Layer tests passed!" << std::endl;
}
