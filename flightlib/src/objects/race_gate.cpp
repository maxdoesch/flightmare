#include "flightlib/objects/race_gate.hpp"
#include <iostream>

namespace flightlib {
    Eigen::Matrix<float, 3, 4> RaceGate::getCorners()
    {
        float width = getSize()[0];
        float height = getSize()[2];
        
        Eigen::Matrix<float, 3, 4> localCorners;
        localCorners << width/2, -width/2, -width/2, width/2,
                        0, 0, 0, 0,
                        height/2, height/2, -height/2, -height/2;

        Eigen::Vector3f translation = getPosition();
        Eigen::Matrix3f rotation = getQuaternion().toRotationMatrix();

        //std::cout << "local" << std::endl;
        //std::cout << localCorners << std::endl;
        //std::cout << "rotated" << std::endl;
        //std::cout << rotation * localCorners << std::endl;
        //std::cout << "----" << std::endl;
        
        return (rotation * localCorners).colwise() + translation;
    }
}