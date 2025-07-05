import smbus2
from util_pca9685 import list_serial_ports

ports = list_serial_ports()  # 列出可用的串口设备
print(ports)

# # 初始化I2C总线（CH341T通常映射为/dev/i2c-*或通过虚拟串口）
# bus = smbus2.SMBus(0)  # 参数可能需根据系统调整
# device_address = 0x50  # 目标I2C设备地址（如AT24C02 EEPROM）

# # 写入数据（示例：向地址0x00写入字节0xAB）
# bus.write_byte_data(device_address, 0x00, 0xAB)

# # 读取数据（示例：从地址0x00读取1字节）
# data = bus.read_byte_data(device_address, 0x00)
# print(f"Read data: {hex(data)}")
