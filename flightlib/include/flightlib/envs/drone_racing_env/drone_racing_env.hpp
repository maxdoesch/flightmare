#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/race_gate.hpp"

namespace flightlib {

namespace droneracingenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kLinVel = 0,
  kNLinVel = 3,
  kOri = 3,
  kNOri = 9,
  kGate1 = 12,
  kNGate1 = 12,
  kGate2 = 24,
  kNGate2 = 12,
  kNObs = 36,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};
class DroneRacingEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DroneRacingEnv();
  DroneRacingEnv(const std::string &cfg_path);
  ~DroneRacingEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool train = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;
  bool getGatePassed();
  void get_state(Ref<Vector<>> state);

  void set_initial_states(std::shared_ptr<MatrixRowMajor<>> initial_states);

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  void updateExtraInfo() override;
  bool isTruncated() override;

  friend std::ostream &operator<<(std::ostream &os,
                                  const DroneRacingEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"DroneRacingEnv"};

  // Define reward for training
  Scalar ang_vel_coeff_, crash_penalty_;

  // observations and actions (for RL)
  Vector<droneracingenv::kNObs> quad_obs_;
  Vector<droneracingenv::kNAct> quad_act_;

  // action and observation normalization (for learning)
  Vector<droneracingenv::kNAct> act_mean_;
  Vector<droneracingenv::kNAct> act_std_;
  Vector<droneracingenv::kNObs> obs_mean_ = Vector<droneracingenv::kNObs>::Zero();
  Vector<droneracingenv::kNObs> obs_std_ = Vector<droneracingenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;

  std::shared_ptr<MatrixRowMajor<>> initial_states = NULL;

  std::vector<std::shared_ptr<RaceGate>> gates;
  int next_gate_idx = 0;
  int lap_cnt = 0;

  const int laps_per_race = 3; 

  bool is_truncated = false;
};

}  // namespace flightlib