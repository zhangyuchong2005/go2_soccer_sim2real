/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GO2_FSM_HPP
#define GO2_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"
#include <cstdlib>
#include <iomanip>

namespace go2_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 8;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };
    bool stand_from_passive = true;

    void Enter() override
    {
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        int num_joints = rl.params.Get<int>("num_of_dofs");
        auto joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");

        // Reorder default_dof_pos from policy order back to SDK order for interpolation.
        // motor_state.q and Interpolate both work in SDK order, but default_dof_pos
        // in many policies (e.g. tripod_balance, hello_tripod) uses IsaacLab order.
        bool mapping_is_identity = true;
        for (int i = 0; i < num_joints; ++i) {
            if (joint_mapping[i] != i) { mapping_is_identity = false; break; }
        }

        std::vector<float> default_dof_sdk;
        if (!mapping_is_identity) {
            default_dof_sdk.resize(num_joints);
            for (int i = 0; i < num_joints; ++i) {
                default_dof_sdk[joint_mapping[i]] = rl.params.Get<std::vector<float>>("default_dof_pos")[i];
            }
        } else {
            default_dof_sdk = rl.params.Get<std::vector<float>>("default_dof_pos");
        }

        if(stand_from_passive)
        {

            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            if (Interpolate(percent_getup, pre_running_pos, default_dof_sdk, 2.0f, "Getting up", true)) return;
        }
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, default_dof_sdk, 1.0f, "Getting up", true)) return;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRLLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
            {
                return "RLFSMStateSoccerKick";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        const char* config_name = std::getenv("RL_SAR_GO2_CONFIG");
        rl.config_name = (config_name && config_name[0] != '\0') ? config_name : "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        int num_joints = rl.params.Get<int>("num_of_dofs");
        auto joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");

        // Same reordering for policy transition
        bool mapping_is_identity = true;
        for (int i = 0; i < num_joints; ++i) {
            if (joint_mapping[i] != i) { mapping_is_identity = false; break; }
        }
        std::vector<float> default_dof_sdk;
        if (!mapping_is_identity) {
            default_dof_sdk.resize(num_joints);
            for (int i = 0; i < num_joints; ++i) {
                default_dof_sdk[joint_mapping[i]] = rl.params.Get<std::vector<float>>("default_dof_pos")[i];
            }
        } else {
            default_dof_sdk = rl.params.Get<std::vector<float>>("default_dof_pos");
        }

        // position transition from last default_dof_pos to current default_dof_pos
        if (Interpolate(
                percent_transition,
                rl.now_state.motor_state.q,
                default_dof_sdk,
                rl.params.Get<float>("policy_transition_duration_s", 1.0f),
                "Policy transition",
                true))
        {
            return;
        }

        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateSoccerKick";
        }
        return state_name_;
    }
};

class RLFSMStateSoccerKick : public RLFSMState
{
public:
    RLFSMStateSoccerKick(RL *rl) : RLFSMState(*rl, "RLFSMStateSoccerKick") {}

    float elapsed_time = 0.0f;
    float kick_duration = 1.0f;
    float auto_return_time = 1.15f;
    float safety_scale = 1.0f;
    float kick_kp = 45.0f;
    float kick_kd = 5.0f;
    bool fr_hip_locked = true;
    float roll_threshold = 30.0f;
    float pitch_threshold = 30.0f;

    // FR leg keyframes (SDK order: hip, thigh, calf)
    std::vector<float> fr_backswing = {0.0f, 1.35f, -2.35f};
    std::vector<float> fr_strike    = {0.0f, -0.28f, -2.05f};
    std::vector<float> fr_follow    = {0.0f, -0.34f, -1.95f};

    // Phase boundaries: [backswing_end, strike_end, follow_end]
    std::vector<float> phase_boundaries = {0.35f, 0.65f, 0.80f};

    // Standing pose captured on entry (SDK order, 12 joints)
    std::vector<float> standing_pose;

    void Enter() override
    {
        elapsed_time = 0.0f;

        // Load soccer_kick config (ReadYaml only, no model needed)
        rl.config_name = "soccer_kick";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        rl.ReadYaml(robot_config_path, "config.yaml");

        // Read config parameters
        kick_duration = rl.params.Get<float>("kick_duration", 1.0f);
        auto_return_time = rl.params.Get<float>("auto_return_time", 1.15f);
        safety_scale = rl.params.Get<float>("safety_scale", 1.0f);
        kick_kp = rl.params.Get<float>("kick_kp", 45.0f);
        kick_kd = rl.params.Get<float>("kick_kd", 5.0f);
        fr_hip_locked = rl.params.Get<bool>("fr_hip_locked", true);
        roll_threshold = rl.params.Get<float>("roll_threshold", 30.0f);
        pitch_threshold = rl.params.Get<float>("pitch_threshold", 30.0f);

        fr_backswing = rl.params.Get<std::vector<float>>("fr_backswing", {0.0f, 1.35f, -2.35f});
        fr_strike    = rl.params.Get<std::vector<float>>("fr_strike",    {0.0f, -0.28f, -2.05f});
        fr_follow    = rl.params.Get<std::vector<float>>("fr_follow",    {0.0f, -0.34f, -1.95f});
        phase_boundaries = rl.params.Get<std::vector<float>>("phase_boundaries", {0.35f, 0.65f, 0.80f});

        // Capture current standing pose
        standing_pose = rl.now_state.motor_state.q;

        std::cout << LOGGER::INFO << "SoccerKick started: duration=" << kick_duration
                  << "s safety_scale=" << safety_scale << std::endl;
    }

    float smoothstep(float x) const
    {
        x = std::max(0.0f, std::min(1.0f, x));
        return x * x * (3.0f - 2.0f * x);
    }

    void Run() override
    {
        int num_joints = rl.params.Get<int>("num_of_dofs");
        auto joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");

        // Attitude protect every step
        rl.AttitudeProtect(rl.now_state.imu.quaternion, pitch_threshold, roll_threshold);

        // Compute phase (0 to 1 over kick_duration)
        float phase = std::min(1.0f, elapsed_time / kick_duration);

        // Compute FR leg target from keyframes
        float b1 = phase_boundaries[0];  // backswing end
        float b2 = phase_boundaries[1];  // strike end
        float b3 = phase_boundaries[2];  // follow end

        std::vector<float> fr_target(3);
        if (phase < b1) {
            float u = smoothstep(phase / b1);
            for (int i = 0; i < 3; i++) fr_target[i] = (1.0f - u) * standing_pose[i] + u * fr_backswing[i];
        } else if (phase < b2) {
            float u = smoothstep((phase - b1) / (b2 - b1));
            for (int i = 0; i < 3; i++) fr_target[i] = (1.0f - u) * fr_backswing[i] + u * fr_strike[i];
        } else if (phase < b3) {
            float u = smoothstep((phase - b2) / (b3 - b2));
            for (int i = 0; i < 3; i++) fr_target[i] = (1.0f - u) * fr_strike[i] + u * fr_follow[i];
        } else {
            float u = smoothstep((phase - b3) / (1.0f - b3));
            for (int i = 0; i < 3; i++) fr_target[i] = (1.0f - u) * fr_follow[i] + u * standing_pose[i];
        }

        // Apply safety_scale: scale deviation from standing pose
        for (int i = 0; i < 3; i++) {
            fr_target[i] = standing_pose[i] + safety_scale * (fr_target[i] - standing_pose[i]);
        }

        // Lock FR hip to measured standing angle if configured
        if (fr_hip_locked) {
            fr_target[0] = standing_pose[0];
        }

        // Build full 12-joint target: FR leg gets keyframe, others stay at standing
        // SDK order: [0-2]=FR, [3-5]=FL, [6-8]=RR, [9-11]=RL
        std::vector<float> target_q = standing_pose;
        target_q[0] = fr_target[0];  // FR_hip
        target_q[1] = fr_target[1];  // FR_thigh
        target_q[2] = fr_target[2];  // FR_calf

        // Apply safety_scale to non-FR legs too (small deviation to keep body stable)
        for (int i = 3; i < num_joints; i++) {
            target_q[i] = standing_pose[i];
        }

        // Set PD command
        for (int i = 0; i < num_joints; i++) {
            fsm_command->motor_command.q[i] = target_q[i];
            fsm_command->motor_command.dq[i] = 0.0f;
            fsm_command->motor_command.kp[i] = kick_kp;
            fsm_command->motor_command.kd[i] = kick_kd;
            fsm_command->motor_command.tau[i] = 0.0f;
        }

        elapsed_time += rl.params.Get<float>("dt", 0.005f) * rl.params.Get<int>("decimation", 4);

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "SoccerKick phase="
                  << std::fixed << std::setprecision(2) << phase
                  << " t=" << elapsed_time << "s" << std::flush;
    }

    void Exit() override
    {
        std::cout << std::endl;
    }

    std::string CheckChange() override
    {
        // Emergency passive
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        // Auto return to GetUp after kick completes
        if (elapsed_time >= auto_return_time)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

} // namespace go2_fsm

class Go2FSMFactory : public FSMFactory
{
public:
    Go2FSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<go2_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<go2_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<go2_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<go2_fsm::RLFSMStateRLLocomotion>(rl);
        else if (state_name == "RLFSMStateSoccerKick")
            return std::make_shared<go2_fsm::RLFSMStateSoccerKick>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "go2"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion",
            "RLFSMStateSoccerKick"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(Go2FSMFactory, "RLFSMStatePassive")

#endif // GO2_FSM_HPP
