#pragma once
#include <array>
#include <string_view>

using namespace std::string_view_literals;

namespace operational_space_controller::constants {
    namespace model {
        // Mujoco Model Constants:
        constexpr int nq_size  = 15;
        constexpr int nv_size = 14;
        constexpr int nu_size  = 8;
        constexpr int body_ids_size = 9;
        constexpr int site_ids_size = 9;
        constexpr int noncontact_site_ids_size = 1;
        constexpr int contact_site_ids_size = 8;
        constexpr std::array body_list = {"torso"sv, "torso_left_front_wheel"sv, "torso_left_rear_wheel"sv, "torso_right_front_wheel"sv, "torso_right_rear_wheel"sv, "head_left_front_wheel"sv, "head_left_rear_wheel"sv, "head_right_front_wheel"sv, "head_right_rear_wheel"sv};
        constexpr std::array site_list = {"torso_site"sv, "tlf_wheel_site"sv, "tlr_wheel_site"sv, "trf_wheel_site"sv, "trr_wheel_site"sv, "hlf_wheel_site"sv, "hlr_wheel_site"sv, "hrf_wheel_site"sv, "hrr_wheel_site"sv};
        constexpr std::array noncontact_site_list = {"torso_site"sv};
        constexpr std::array contact_site_list = {"tlf_wheel_site"sv, "tlr_wheel_site"sv, "trf_wheel_site"sv, "trr_wheel_site"sv, "hlf_wheel_site"sv, "hlr_wheel_site"sv, "hrf_wheel_site"sv, "hrr_wheel_site"sv};
    }
    namespace optimization {
        // Optimization Constants:
        constexpr int dv_size = 14;
        constexpr int u_size = 8;
        constexpr int z_size = 24;
        constexpr int design_vector_size = 46;
        constexpr int dv_idx = 14;
        constexpr int u_idx = 22;
        constexpr int z_idx = 46;
        constexpr int beq_sz = 14;
        constexpr int Aeq_sz = 644;
        constexpr int Aeq_rows = 14;
        constexpr int Aeq_cols = 46;
        constexpr int bineq_sz = 32;
        constexpr int Aineq_sz = 1472;
        constexpr int Aineq_rows = 32;
        constexpr int Aineq_cols = 46;
        constexpr int H_sz = 2116;
        constexpr int H_rows = 46;
        constexpr int H_cols = 46;
        constexpr int f_sz = 46;
    }
}
        