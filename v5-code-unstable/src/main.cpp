#include "main.h"
#include "pros/apix.h"
#include <array>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

// Detection struct as before
struct Detection {
    double cx, cy, w, h;
    int cls_index;
    double confidence, depth;
};

// Frame struct as before
struct Frame {
    array<pair<double, double>, 10> loc;
    vector<Detection> det = {};
    Frame() {
        while (true) {
            char t = cin.get();
            if (t == 'b') {
                for (auto &[x, y] : loc) {
                    cin >> x >> y;
                }
            } else if (t == 'a') {
                double cx, cy, w, h;
                int cls_index;
                double confidence, depth;
                cin >> cx >> cy >> w >> h >> cls_index >> confidence >> depth;
                det.push_back(
                    Detection{cx, cy, w, h, cls_index, confidence, depth});
            } else {
                break;
            }
        }
    }
};

namespace serial {
    void activate() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);
        cout << "activate" << endl;
    }
} // namespace serial

void initialize() {}

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {
    serial::activate();

    lv_obj_t *screen = lv_scr_act();

    // We'll keep track of the last set of rectangles so we can clear them
    vector<lv_obj_t *> last_rects;

    while (true) {
        Frame frame;

        if (!frame.det.empty()) {
            cout << "a" << endl;
            // Clear previous rectangles
            for (auto *rect : last_rects) {
                lv_obj_del(rect);
            }
            last_rects.clear();

            // Draw new rectangles for each detection
            for (const auto &det : frame.det) {
                int x = static_cast<int>(det.cx - det.w / 2);
                int y = static_cast<int>(det.cy - det.h / 2);
                int w = static_cast<int>(det.w);
                int h = static_cast<int>(det.h);

                lv_obj_t *rect = lv_obj_create(screen);
                lv_obj_set_size(rect, w, h);
                lv_obj_set_pos(rect, x, y);

                static lv_style_t style;
                static bool style_initialized = false;
                if (!style_initialized) {
                    lv_style_init(&style);
                    lv_style_set_bg_opa(&style, LV_OPA_TRANSP);
                    lv_style_set_border_color(&style, lv_color_hex(0xFF0000));
                    lv_style_set_border_width(&style, 2);
                    style_initialized = true;
                }
                lv_obj_add_style(rect, &style, 0);

                last_rects.push_back(rect);
            }
        }
        // else: do nothing, keep previous rectangles

        lv_task_handler();
        pros::delay(20);
    }
}
