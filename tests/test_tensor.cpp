#include "cnn/cnn.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace cnn;

void run_tensor_tests() {
    std::cout << "Running Tensor tests..." << std::endl;
    
    // Test basic tensor creation and operations
    Tensor a({2, 3});
    a.fill(2.0f);
    
    Tensor b({2, 3});
    b.fill(3.0f);
    
    auto c = a + b;
    assert(std::abs(c.at(0, 0) - 5.0f) < 1e-6f);
    
    // Test matrix multiplication
    Tensor x({2, 3});
    x.at(0, 0) = 1; x.at(0, 1) = 2; x.at(0, 2) = 3;
    x.at(1, 0) = 4; x.at(1, 1) = 5; x.at(1, 2) = 6;
    
    Tensor y({3, 2});
    y.at(0, 0) = 1; y.at(0, 1) = 2;
    y.at(1, 0) = 3; y.at(1, 1) = 4;
    y.at(2, 0) = 5; y.at(2, 1) = 6;
    
    auto z = x.matmul(y);
    assert(z.shape()[0] == 2 && z.shape()[1] == 2);
    assert(std::abs(z.at(0, 0) - 22.0f) < 1e-6f);  // 1*1 + 2*3 + 3*5 = 22
    
    std::cout << "Tensor tests passed!" << std::endl;
}
