#include "flightlib/envs/drone_racing_env/drone_racing_env.hpp"

namespace flightlib {

DroneRacingEnv::DroneRacingEnv()
  : DroneRacingEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/drone_racing_env.yaml")) {}

DroneRacingEnv::DroneRacingEnv(const std::string &cfg_path)
  : EnvBase(),
    ang_vel_coeff_(0.0) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = droneracingenv::kNObs;
  act_dim_ = droneracingenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<droneracingenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<droneracingenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  extra_info_.insert({"TimeLimit.truncated", false});
  extra_info_.insert({"gate_idx", 0});
  extra_info_.insert({"lap_cnt", 0});
  extra_info_.insert({"is_success", false});

  // load parameters
  loadParam(cfg_);
}

DroneRacingEnv::~DroneRacingEnv() {}

bool DroneRacingEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    // reset position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) + 5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  next_gate_idx = 0;
  lap_cnt = 0;

  // obtain observations
  getObs(obs);
  return true;
}

bool DroneRacingEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  int nextnext_gate_idx = (next_gate_idx + 1) % gates.size();
  Eigen::Matrix<float, 3, 4> next_gate_rel_pos = gates[next_gate_idx]->getCorners().colwise() - quad_state_.p;
  Eigen::Matrix<float, 3, 4> nextnext_gate_rel_pos = gates[nextnext_gate_idx]->getCorners().colwise() - quad_state_.p;

  // convert quaternion to euler angle
  Eigen::Matrix3f quad_rot = quad_state_.q().toRotationMatrix();

  Eigen::Matrix<float, 12, 1> f_next_gate_rel_pos = Eigen::Map<Eigen::Matrix<float, 12, 1>>(next_gate_rel_pos.data());
  Eigen::Matrix<float, 12, 1> f_nextnext_gate_rel_pos = Eigen::Map<Eigen::Matrix<float, 12, 1>>(nextnext_gate_rel_pos.data());
  Eigen::Matrix<float, 9, 1> f_quad_rot = Eigen::Map<Eigen::Matrix<float, 9, 1>>(quad_rot.data());
  quad_obs_ << quad_state_.v, f_quad_rot, f_next_gate_rel_pos, f_nextnext_gate_rel_pos;

  obs.segment<droneracingenv::kNObs>(droneracingenv::kObs) = quad_obs_;
  return true;
}

bool DroneRacingEnv::getGatePassed() {
  quadrotor_ptr_->getState(&quad_state_);

  Eigen::Matrix<float, 3, 4> gate_corners = gates[next_gate_idx]->getCorners();
  Eigen::Matrix<float, 3, 4> state_diff = gate_corners.colwise() - quad_state_.p;
  Eigen::Matrix<float, 1, 4> normed_state_diff = state_diff.colwise().squaredNorm();

  float gate_diagonal = (gate_corners.col(0) - gate_corners.col(2)).squaredNorm();

  return (normed_state_diff.array() < gate_diagonal).all();
}

Scalar DroneRacingEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  Eigen::Vector3f prev_quad_position = quad_state_.p;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  //Check whether gate was passed
  if(getGatePassed()) {
    
    next_gate_idx = (next_gate_idx + 1);
    if (!(next_gate_idx % gates.size())) {
      lap_cnt++;
    }
    next_gate_idx %= gates.size();
  }

  // update observations
  getObs(obs);

  // ---------------------- reward function design
  Scalar gate_reward = (gates[next_gate_idx]->getPosition() - prev_quad_position).squaredNorm() - 
    (gates[next_gate_idx]->getPosition() - quad_state_.p).squaredNorm();

  Scalar act_reward = -ang_vel_coeff_ * quad_state_.w.cast<Scalar>().norm();

  Scalar total_reward = gate_reward + act_reward;

  return total_reward;
}

bool DroneRacingEnv::isTerminalState(Scalar &reward) {
  if (lap_cnt >= laps_per_race) {
    reward = 10;
    return true;
  }
  else if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -10;
    return true;
  }
  reward = 0.0;
  return false;
}

bool DroneRacingEnv::isTruncated()
{
  return cmd_.t > max_t_;
}

void DroneRacingEnv::updateExtraInfo()
{
  extra_info_["TimeLimit.truncated"] = (float) isTruncated();
  extra_info_["gate_idx"] = next_gate_idx;
  extra_info_["lap_cnt"] = lap_cnt;
  extra_info_["is_success"] = (float) (lap_cnt >= laps_per_race);
}

bool DroneRacingEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["race_track"]) {
    //load race track gate positions
    for (const auto& gate : cfg["race_track"]) {
        std::string gate_name = gate.first.as<std::string>();
        std::vector<float> coordinates = gate.second.as<std::vector<float>>();
        Eigen::Vector3f gate_position(coordinates[0], coordinates[1], coordinates[2]);
        float gate_angle = coordinates[3] * M_PI / 180.;

        std::shared_ptr<RaceGate> race_gate = std::make_shared<RaceGate>(gate_name, "rpg_gate");
        race_gate->setPosition(gate_position);
        race_gate->setQuaternion(Quaternion(std::cos(gate_angle / 2.), 0.0, 0.0, std::sin(gate_angle / 2.)));

        //std::cout << gate_position.transpose() << std::endl;
        //std::cout << gate_angle << std::endl;
        //std::cout << "------------" << std::endl;
        //std::cout << race_gate->getCorners() << std::endl;
        //std::cout << "-----------" << std::endl;
        
        gates.push_back(race_gate);
    }
  }
  return true;
}

bool DroneRacingEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool DroneRacingEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void DroneRacingEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);

  for(const auto& gate : gates)
  {
    bridge->addStaticObject(gate);
  }
}

std::ostream &operator<<(std::ostream &os, const DroneRacingEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib