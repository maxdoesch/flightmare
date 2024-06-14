#pragma once

#include "flightlib/objects/static_object.hpp"

namespace flightlib {
class RaceGate : public StaticObject {
 public:
  RaceGate(const std::string& id, const std::string& prefab_id = "rpg_gate")
    : StaticObject(id, prefab_id) {}
  ~RaceGate() {}

  Eigen::Matrix<float, 3, 4> getCorners();
};

}  // namespace flightlib
