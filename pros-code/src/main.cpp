#include "main.h"
#include "pros/apix.h" // Include LVGL access
#include "receiver.hpp"
#include <cstdio>
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

        printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", ix, iy, theta);
        fflush(stdout);
    }

    void task() {
        while (true) {
            // TODO: get the actual imu reading from the imu
            double imu_reading = 5.5;
            printf("{\"imu\": %f}\r\n", imu_reading);
            fflush(stdout);

            most_recent_frame = fetch_frame();

            pros::delay(20);
        }
    }

} // namespace serial

void initialize() {
    // IMPORTANT: Remove or comment out LLEMU initialization if present
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {
    serial::start();
    pros::Task serial_task(serial::task);

}
