#include "pros/apix.h" // Include for serctl
#include "receiver.hpp"
#include <cstdio>
#include "robot.hpp"
using namespace robot::chassis;
namespace serial {
    void task() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);
        while (true) {
            // TODO: get the actual imu reading from the imu
            auto pose = chassis.getPose();
            printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", pose.x, pose.y, pose.theta);
            fflush(stdout);

            most_recent_frame = fetch_frame();

            pros::delay(20);
        }
    }
} // namespace serial
