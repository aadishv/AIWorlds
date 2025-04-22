#include "main.h"
#include "nlohmann/json.hpp"
#include "pros/apix.h"
#include "schema.hpp"
#include <array>
#include <iostream>
#include <utility>
#include <vector>

using namespace nlohmann;
using namespace std;
// Frame struct as before
struct Frame {
    vector<pair<double, double>> loc;
    vector<Detection> det = {};
};

namespace serial {
    void activate() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);
        cout << "activate" << endl;
    }
    std::optional<Frame> get_next_frame() {
        try {
            json j;
            cin >> j;

            if (cin.fail()) {
                return std::nullopt;
            }

            // get points
            vector<pair<double, double>> v(10);

            if (!j.contains("points") || !j["points"].is_array() ||
                j["points"].size() < 20) {
                return std::nullopt;
            }

            int i = 0;
            for (auto &p : v) {
                p = make_pair(j["points"][i], j["points"][i + 1]);
                i += 2;
            }

            // get detected objects
            vector<Detection> dets;
            if (j.contains("dets") && j["dets"].is_array()) {
                for (auto det : j["dets"]) {
                    try {
                        dets.push_back(Detection::from_json(det));
                    } catch (...) {
                        // Skip this detection if it fails to parse
                        continue;
                    }
                }
            }

            return Frame{v, dets};
        } catch (...) {
            return std::nullopt;
        }
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
        optional<Frame> frame = serial::get_next_frame();
        if (!frame.has_value())
            continue;
        // Clear previous rectangles
        for (auto *rect : last_rects) {
            lv_obj_del(rect);
        }
        last_rects.clear();

        // Draw new rectangles for each detection
        for (const auto &det : frame.value().det) {
            int x = static_cast<int>(det.x - det.width / 2);
            int y = static_cast<int>(det.y - det.height / 2);
            int w = static_cast<int>(det.width);
            int h = static_cast<int>(det.height);

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

        lv_task_handler();
        pros::delay(20);
    }
}
