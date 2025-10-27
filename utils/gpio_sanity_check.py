import time, Jetson.GPIO as GPIO
from pytictoc import TicToc
# GPIO.setmode(GPIO.BOARD)        # use physical pin numbers
# PINS = (12, 15, 16 ,18, 22, 24, 26)             #

GPIO.setmode(GPIO.BCM)        # use physical pin numbers
PINS = (18, 19, 21, 22)             #


for p in PINS:
    GPIO.setup(p, GPIO.IN)      # external pull-ups/downs required

def read_idx():
    a = 1 if GPIO.input(22) else 0
    b = 1 if GPIO.input(24) else 0
    c = 1 if GPIO.input(16) else 0
    if a or b or c:
        print(f"A={a}, B={b}, C={c}")
        print(f"16: GPIO.input(22): {GPIO.input(22)}, "
              f"18: GPIO.input(24): {GPIO.input(24)}, "
              f"15: GPIO.input(16): {GPIO.input(16)}")
    return (c << 2) | (b << 1) | a, (a,b,c)

def get_trigger():
    trigger = GPIO.input(18)
    return 1 if trigger else 0

def get_LSB():
    return GPIO.input(19)

def get_middle():
    return GPIO.input(22)

def get_MSB():
    return GPIO.input(21)

# print("Reading 12/22,24,16 (BOARD). Ctrl+C to exit.")
try:
    prev_value = None
    lsb_prev_value = None
    middle_prev_value = None
    msb_prev_value = None
    t_lsb = TicToc()  # Create an instance
    t_middle = TicToc()
    t_msb = TicToc()
    t_trigger = TicToc()
    while True:
        lsb = get_LSB()
        if lsb != lsb_prev_value:
            lsb_prev_value = lsb
            if lsb == GPIO.HIGH:
                t_lsb.tic()  # Start timer
                print("LSB is HIGH")
            else:
                print("LSB is LOW")
                t_lsb.toc("LSB tic toc")
        middle = get_middle()
        if middle != middle_prev_value:
            middle_prev_value = middle
            t_middle.tic()
            if middle == GPIO.HIGH:
                print("Middle is HIGH")
                t_middle.tic()
            else:
                print("Middle is LOW")
                t_middle.toc("Middle tic toc")
        msb = get_MSB()
        if msb != msb_prev_value:

            msb_prev_value = msb
            if msb == GPIO.HIGH:
                print("MSB is HIGH")
                t_msb.tic("MSB tic start")
            else:
                print("MSB is LOW")
                t_msb.toc("MSB toc")

        # trigger = get_trigger()
        # if trigger != prev_value:
        #     if trigger == GPIO.HIGH:
        #         t_trigger.tic()
        #         value_str = "HIGH"
        #     else:
        #         value_str = "LOW"
        #         t_trigger.toc("trigger tic toc")
        #     print("Value read from pin {} : {}".format(12,
        #                                                value_str))
        #     prev_value = trigger
        # else:
        #     print(f"prev = {prev_value}, trigger = {trigger}")
        time.sleep(0.1)
finally:
    GPIO.cleanup()
