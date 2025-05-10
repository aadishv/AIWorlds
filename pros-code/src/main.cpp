#include "main.h"
#include "jetson.hpp"

#include <cstdio>
using namespace std;
void initialize() {
    // Initialize LVGL
    pros::lcd::initialize();
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {

    // Create serial task
    pros::Task serial_task_send(serial::task_send);
    pros::Task serial_task_receive(serial::task_receive);

    // Create visualizer task
    pros::Task visualizer_task(visualizer_task_fn);

    // LVGL visualization
     // Create a label to display detection count
    lv_obj_t* label = lv_label_create(lv_scr_act());
    lv_label_set_text(label, "Detections: 0");
    lv_obj_align(label, LV_ALIGN_CENTER, 0, 0);

    // Set font size (larger text)
    lv_obj_set_style_text_font(label, &lv_font_montserrat_24, 0);

    while (true) {
        // Update display with number of detections
        if (serial::most_recent_frame) {
            int detection_count = serial::most_recent_frame->detections.size();
            char buffer[50];
            snprintf(buffer, sizeof(buffer), "Detections: %d", detection_count);
            lv_label_set_text(label, buffer);
        } else {
            lv_label_set_text(label, "Detections: 0");
        }

        // Refresh every 100ms
        pros::delay(100);
    }
}
