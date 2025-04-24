#include "main.h"
#include "pros/apix.h" // Include LVGL access
#include "receiver.hpp"
#include <cstdio>     // Include for snprintf

using namespace std;

// Forward declaration for the visualizer task
void visualizer_task();

namespace serial {
    using namespace serial;
    // Make most_recent_frame accessible (it already is globally in the namespace)
    optional<Frame> most_recent_frame = nullopt;

    void start() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);
        // TODO: get the actual starting position from the auton selector or smth
        double ix = 1.0;
        double iy = 2.0;
        double theta = 180.0;

        printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", ix, iy, theta); // Added \r\n for clarity
        fflush(stdout);
    }

    void task() {
        while (true) {
            // TODO: get the actual imu reading from the imu
            double imu_reading = 5.5;
            printf("{\"imu\": %f}\r\n", imu_reading); // Added \r\n for clarity
            fflush(stdout);

            most_recent_frame = fetch_frame();

            pros::delay(20);
        }
    }

} // namespace serial

// --- Visualizer Task ---
void visualizer_task() {
    while (true) {
        lv_obj_clean(lv_scr_act());

        // NOTE: This is NOT thread-safe. In a real application, use a mutex
        // or other synchronization mechanism to protect access between
        // serial::task and visualizer_task. Per your instructions, ignoring this for now.
        if (serial::most_recent_frame.has_value()) {
            const auto& frame = serial::most_recent_frame.value();

            for (const auto& detection : frame.detections) {
                lv_color_t rect_color;
                if (detection.cls == "red") {
                    rect_color = lv_color_hex(0xFF0000);
                } else if (detection.cls == "blue") {
                    rect_color = lv_color_hex(0x0000FF);
                } else if (detection.cls == "goal") {
                    rect_color = lv_color_hex(0xFFFF00);
                } else if (detection.cls == "robot") {
                    rect_color = lv_color_hex(0x00FF00);
                } else {
                    rect_color = lv_color_hex(0x808080);
                }

                lv_obj_t* rect = lv_obj_create(lv_scr_act());
                lv_obj_remove_style_all(rect);

                lv_obj_set_pos(rect, static_cast<lv_coord_t>(detection.x), static_cast<lv_coord_t>(detection.y + 20));
                lv_obj_set_size(rect, static_cast<lv_coord_t>(detection.w), static_cast<lv_coord_t>(detection.h));

                lv_obj_set_style_bg_color(rect, rect_color, 0);
                lv_obj_set_style_bg_opa(rect, LV_OPA_50, 0);
                lv_obj_set_style_border_color(rect, rect_color, 0);
                lv_obj_set_style_border_width(rect, 2, 0);
                lv_obj_set_style_border_opa(rect, LV_OPA_COVER, 0);

                lv_obj_t* label = lv_label_create(rect);

                char conf_str[10];
                snprintf(conf_str, sizeof(conf_str), "%.0f%%", detection.conf * 100.0);
                lv_label_set_text(label, conf_str);

                lv_obj_set_style_text_color(label, lv_color_white(), 0);
                lv_obj_set_style_text_opa(label, LV_OPA_COVER, 0);

                lv_obj_align(label, LV_ALIGN_CENTER, 0, 0);
            }
        }

        pros::delay(1000 / 30);
    }
}
// --- End Visualizer Task ---


void initialize() {
    // IMPORTANT: Remove or comment out LLEMU initialization if present
    // pros::lcd::initialize(); // REMOVE THIS LINE if it exists
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {
    serial::start();
    pros::Task serial_task(serial::task);

    pros::Task screen_task(visualizer_task); // Renamed handle for clarity

    // Keep opcontrol alive (optional, tasks run independently)
    // while (true) {
    //     // Main opcontrol loop logic (e.g., driver control) can go here
    //     pros::delay(20);
    // }
}
