import serial
import time

# 配置串口参数
# 请根据实际情况修改以下参数
serial_port = 'COM18'  # 串口号，例如'COM3'或'/dev/ttyUSB0'
baud_rate = 115200      # 波特率，需要与STM32程序中设置的波特率一致
timeout = 1           # 读超时设置

# 初始化串口
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
    print(f"串口{serial_port}已打开。")
except serial.SerialException as e:
    print(f"无法打开串口{serial_port}：{e}")
    exit()

# 确保串口已打开
if ser.is_open:
    try:
        # 发送数据到STM32
        command = 'HELLO STM32' + '\r\n'  # 假设STM32在接收到换行符后开始处理数据
        ser.write(command.encode("utf-8"))  # 发送数据前需要编码为字节
        time.sleep(0.1)
        ser.write(command.encode("utf-8"))
        time.sleep(0.1)
        ser.write(command.encode("utf-8"))
        time.sleep(0.1)
        
        print(f"发送数据：{command}")

        # 稍作延时，等待STM32处理并返回数据
        time.sleep(5)

        # 读取STM32返回的数据
        while ser.in_waiting > 0:
            response = ser.readline().decode().strip()  # 读取一行数据，并解码去除尾部的换行符
            print(f"接收到数据：{response}")

    except serial.SerialException as e:
        print(f"串口通信出错：{e}")
    finally:
        # 关闭串口
        ser.close()
        print(f"串口{serial_port}已关闭。")
else:
    print(f"串口{serial_port}未打开。")
