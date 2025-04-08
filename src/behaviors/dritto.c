#include <kilolib.h>

void setup()
{
}

void loop()
{
    // Set the LED green.
    set_color(RGB(0, 1, 0));
    // Spinup the motors to overcome friction.
    // Move straight for 2 seconds (2000 ms).
    spinup_motors();

    set_motors(kilo_straight_left, kilo_straight_right);
    delay(65000);
}

int main()
{
    kilo_init();
    kilo_start(setup, loop);

    return 0;
}
