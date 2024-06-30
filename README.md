```
# Motion Control Script

This script is designed to control the motion of a optics focus motion controller through a serial connection. It uses the parameters specified in the configuration file `config.txt` to perform specific actions, such as returning to the origin, checking the connection, setting the speed, and moving the controller.

## Prerequisites

- Required Python packages: `serial`, `configparser`

## Installation

1. Clone the repository or download the script `motion_control.py` and the configuration file `config.txt`.

2. Install the required Python packages using pip:

   ```bash
   pip install pyserial
   ```

3. Configure the `config.txt` file to set the appropriate values for the serial port, baud rate, motion settings, and other parameters.

## Usage

1. Make sure the device is connected to the specified serial port.

2. Run the script `motion_control.py`:

   ```bash
   python motion_control.py
   ```

3. The script will execute the following actions:

   - Return to the origin before executing the motion control code.
   - Check the connection to the motion controller.
   - Set the speed of the motion controller.
   - Move the controller in the specified direction and displacement value.
   - Take a picture after each movement.
   - Return to the origin after executing the motion control code.

4. The script will print the status and responses for each action to the console.

## Configuration

You can customize the settings in the `config.txt` file to change the behavior of the script.
