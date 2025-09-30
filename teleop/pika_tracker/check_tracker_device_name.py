from pika.sense import Sense
import time

def main():
    
    while True:
        device_id = input('Please enter the device id for checking, examples: /dev/ttyUSB0: ')
        if device_id == 'q':
            print(f'ready to quit!!!!')
            break
        
        sense = Sense(device_id)
        if not sense.connect():
            print(f'Failed to connect {device_id}')
            continue
        
        device_names = sense.get_tracker_devices()
        print(f'tracker device names: {device_names}')
        time.sleep(1.0)


if __name__ == "__main__":
    main()