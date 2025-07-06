from math import floor
from time import sleep
import serial


class PCA9685:
    """Control PCA9685 PWM controller via I2C interface.
    
    Attributes
    ----------
    REGs_LED_ON : list[list[int]]
        Register addresses for LED ON times (16 channels)
    REGs_LED_OFF : list[list[int]] 
        Register addresses for LED OFF times (16 channels)
    REG_FREQ : int
        Register address for PWM frequency prescaler
    REG_MODE1 : int
        Register address for MODE1 configuration
    REG_MODE2 : int
        Register address for MODE2 configuration
    RST_ADR : int
        Reset I2C address
    INT_OSC_FREQ : float
        Internal oscillator frequency (25 MHz)
    LED_LEVELS : int
        Number of PWM levels (4096)
    MIN_DELAY : float
        Minimum I2C bus delay (5 μs)
    """
    REGs_LED_ON = [
        [0x06, 0x07],  # 00
        [0x0a, 0x0b],  # 01
        [0x0e, 0x0f],  # 02
        [0x12, 0x13],  # 03
        [0x16, 0x17],  # 04
        [0x1a, 0x1b],  # 05
        [0x1e, 0x1f],  # 06
        [0x22, 0x23],  # 07
        [0x26, 0x27],  # 08
        [0x2a, 0x2b],  # 09
        [0x2e, 0x2f],  # 10
        [0x32, 0x33],  # 11
        [0x36, 0x37],  # 12
        [0x3a, 0x3b],  # 13
        [0x3e, 0x3f],  # 14
        [0x42, 0x43],  # 15
    ]
    REGs_LED_OFF = [
        [0x08, 0x09],  # 00
        [0x0c, 0x0d],  # 01
        [0x10, 0x11],  # 02
        [0x14, 0x15],  # 03
        [0x18, 0x19],  # 04
        [0x1c, 0x1d],  # 05
        [0x20, 0x21],  # 06
        [0x24, 0x25],  # 07
        [0x28, 0x29],  # 08
        [0x2c, 0x2d],  # 09
        [0x30, 0x31],  # 10
        [0x34, 0x35],  # 11
        [0x38, 0x39],  # 12
        [0x3c, 0x3d],  # 13
        [0x40, 0x41],  # 14
        [0x44, 0x45],  # 15
    ]
    REG_FREQ = 0xfe
    REG_MODE1 = 0x00
    REG_MODE2 = 0x01

    RST_ADR = 0x03
    INT_OSC_FREQ = 25e6  # 25 MHz
    LED_LEVELS = 4096
    MIN_DELAY = 5e-6  # seconds, I2C bus max 100 kHz

    def __init__(self, i2c_address: int = 0x40) -> None:
        """Initialize PCA9685 controller.
        
        Parameters
        ----------
        i2c_address : int, optional
            I2C device address, by default 0x40
            
        Notes
        -----
        - Sets default PWM frequency to 200 Hz
        - Initializes MODE1 and MODE2 registers
        - Uses SerialToI2CSender for communication
        """
        # if board_adr is None:
        #     board_adr = firmata.Arduino.AUTODETECT
        # if board_adr is None:  # still None, then Arduino not connected
        #     self.board = FakeArduino()
        # else:
        #     self.board = firmata.Arduino(board_adr)
        self.freq = 200  # Hz, hardware default
        self.prescale = 30
        self.sender = SerialToI2CSender(i2c_address=i2c_address)
        self.mode1 = {
            "RESTART": False,   # 7
            "EXTCLK": False,    # 6
            "AI": False,        # 5
            "SLEEP": True,      # 4
            "SUB1": False,      # 3
            "SUB2": False,      # 2
            "SUB3": False,      # 1
            "ALLCALL": True,    # 0
        }
        self.mode2 = {
            "INVRT": False,     # 4
            "OCH": False,       # 3
            "OUTDRV": True,     # 2
            "OUTNE": [False, False],  # 1, 0
        }
        self.subadr1 = 0xe2
        self.subadr2 = 0xe4
        self.subadr3 = 0xe8
        self.allcalladr = 0xe0
        self.led_levels = None
    
    @property
    def i2c_address(self):
        return self.sender.i2c_address

    @i2c_address.setter
    def i2c_address(self, value: int):
        """Set the I2C address for the sender."""
        self.sender.i2c_address = value

    @staticmethod
    def int2bit_list(x: int, min_len: int) -> list[int]:
        """Convert integer to bit list with specified minimum length.
        
        Parameters
        ----------
        x : int
            Integer to convert (must be >= 0)
        min_len : int
            Minimum length of output bit list
            
        Returns
        -------
        list[int]
            Bit list (MSB first) padded with zeros to min_len
            
        Raises
        ------
        AssertionError
            If x is negative
        """
        assert x >= 0
        bin_str = bin(x)[2:]
        bit_list = [int(ch) for ch in reversed(bin_str)]
        if len(bit_list) < min_len:
            bit_list += [0, ] * (min_len - len(bit_list))
        bit_list.reverse()  # convert low bit first to high bit first
        return bit_list
    
    @staticmethod
    def bits2byte(bit_list: list[int]) -> bytes:
        """Convert bit list to single byte.
        
        Parameters
        ----------
        bit_list : list[int]
            List of bits (length <= 8)
            
        Returns
        -------
        bytes
            Single byte containing the bits (LSB first)
        """
        b = sum(bit * (2 ** i) for i, bit in enumerate(reversed(bit_list)))
        return b.to_bytes(1, byteorder='little')

    def led_level_to_on_off_bytes(self, led_level: float, offset: float = 0.) -> tuple:
        """Convert LED level to ON/OFF register values.
        
        Parameters
        ----------
        led_level : float
            PWM duty cycle (0.0 to 1.0)
        offset : float, optional
            Phase offset (0.0 to 1.0), by default 0.0
            
        Returns
        -------
        tuple
            ((on_low_bits, on_high_bits), (off_low_bits, off_high_bits))
            
        Raises
        ------
        ValueError
            If led_level is outside valid range
        """
        assert 0. <= led_level <= 1.
        offset = offset % 1.
        on_int = floor(0.5 + self.LED_LEVELS * offset)
        led_level_int = min(floor(0.5 + self.LED_LEVELS * led_level), self.LED_LEVELS - 1)
        if 0 <= led_level_int <= self.LED_LEVELS:
            on_bit_list = self.int2bit_list(on_int, min_len=16)
            on_bits_high = on_bit_list[:8]
            on_bits_low = on_bit_list[8:]
            off_int = (on_int + led_level_int) % self.LED_LEVELS
            off_bit_list = self.int2bit_list(off_int, min_len=16)
            off_bits_high = off_bit_list[:8]
            off_bits_low = off_bit_list[8:]
        # elif led_level_int == 0:
        #     on_bits_high = [0, ] * 8
        #     on_bits_low = [0, ] * 8
        #     off_int = (on_int + led_level_int) % self.LED_LEVELS
        #     off_bit_list = self.int2bit_list(off_int, min_len=16)
        #     off_bits_high = off_bit_list[:8]
        #     off_bits_low = off_bit_list[8:]
        #     off_bits_high[3] = 1  # full off
        # elif led_level_int == self.LED_LEVELS:
        #     on_bit_list = self.int2bit_list(on_int, min_len=16)
        #     on_bits_high = on_bit_list[:8]
        #     on_bits_low = on_bit_list[8:]
        #     on_bits_high[3] = 1  # full on
        #     off_bits_high = [0, ] * 8
        #     off_bits_low = [0, ] * 8
        else:
            raise ValueError("Internal error: led_level_int out of range.")
        # on_bits_high = [b for b in reversed(on_bits_high)]
        # on_bits_low = [b for b in reversed(on_bits_low)]
        # off_bits_high = [b for b in reversed(off_bits_high)]
        # off_bits_low = [b for b in reversed(off_bits_low)]
        return (on_bits_low, on_bits_high), (off_bits_low, off_bits_high)

    def set_channels(self, channel_flux_ratios: list[float]) -> None:
        """Set PWM levels for all channels.
        
        Parameters
        ----------
        channel_flux_ratios : list[float]
            List of duty cycles (0.0 to 1.0) for each channel
            
        Notes
        -----
        - Converts each ratio to ON/OFF register values
        - Sends combined data via I2C
        - Currently supports up to 16 channels
        """
        # assert len(channel_flux_ratios) == len(self.REGs_LED_ON)
        # self.send_wake(device_id=device_id)

        # self.mode1["AI"] = True  # Auto Increment (register address)

        # buff = self.bits2byte(self.mode1_to_byte())
        # buff += self.bits2byte(self.mode2_to_byte())
        # buff += bytes([self.subadr1])
        # buff += bytes([self.subadr2])
        # buff += bytes([self.subadr3])
        # buff += bytes([self.allcalladr])
        buff = bytes()

        offset = 0.
        for led_level in channel_flux_ratios:
            # offset = 0.
            on_bytes, off_bytes = self.led_level_to_on_off_bytes(led_level=led_level, offset=offset)
            offset += led_level
            buff += self.bits2byte(on_bytes[0])
            buff += self.bits2byte(on_bytes[1])
            buff += self.bits2byte(off_bytes[0])
            buff += self.bits2byte(off_bytes[1])

        self.sender.send(data=buff)


class SerialToI2CSender:
    """Send I2C commands via serial-to-I2C bridge.
    
    Attributes
    ----------
    port : str
        Serial port name
    i2c_address : int
        I2C device address
    ser : serial.Serial
        Serial connection instance
    """
    
    def __init__(self, i2c_address: int = 0x40) -> None:
        """Initialize serial-to-I2C sender.
        
        Parameters
        ----------
        i2c_address : int, optional
            I2C device address, by default 0x40
            
        Notes
        -----
        - Automatically detects USB serial ports
        - Configures serial connection at 19200 baud
        - Waits 1 second for connection stabilization
        """
        # Configure the serial connection
        ports = list_serial_ports()
        port = None
        for p in ports:
            if "usbserial" in p:
                port = p
                break
        else:
            print(f"No serial ports found. Available ports: {ports}")
        self.port = port
        self.i2c_address = i2c_address
        print(f"Using {port}")
        ser = serial.Serial(self.port, 9600)
        self.ser = ser
        # Give some time to establish the connection
        sleep(1)

    def send(self, data: bytes) -> None:
        """Send I2C data via serial bridge.
        
        Parameters
        ----------
        data : bytes
            Raw I2C data to send
            
        Notes
        -----
        - Prepends I2C address and data length
        - Adds padding bytes
        - Waits proportionally to data length
        """
        # device_id is either 1 or 2
        # Example I2C address and data to send
        # i2c_address = 0x40  # I2C address of the device
        len_data = len(data) // 4

        # Send the I2C address and the number of bytes
        bytes_to_send = bytes([self.i2c_address, len_data]) + data + bytes([0x00, ] * 4)

        # Send the data bytes
        self.ser.flush()
        n_bytes_sent = self.ser.write(bytes_to_send)
        # print(f"Sent {n_bytes_sent} bytes to I2C address 0x{self.i2c_address:02x}:")
        # print("Sent bytes:")
        # print(bytes_to_send.hex(' ', 4))
        sleep(1e-3 * len(bytes_to_send) + 1e-2)

    def stop(self):
        sleep(0.1)
        # Close the serial connection
        self.ser.close()


def list_serial_ports():
    """ Lists serial port names.
        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system.
    """
    from serial.tools import list_ports
    if hasattr(list_ports, 'comports'):
        # Newer versions of pyserial use .comports()
        ports = list_ports.comports()
    else:
        # Fallback for older versions of pyserial
        ports = list_ports.grep('')

    return [port.device for port in ports]


def main():
    print("\n".join(list_serial_ports()))


if __name__ == "__main__":
    main()
