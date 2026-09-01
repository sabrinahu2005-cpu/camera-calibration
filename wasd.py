#!/usr/bin/env python3
# encoding: utf-8
# WASD Keyboard Control for TurboPi Chassis (麥克納姆輪底盤控制)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios
import threading

# 終端機顯示的控制提示
MSG = """
W : 前進
S : 後退
A : 向左平移
D : 向右平移

按SPACE: 煞車停止
"""

# 讀取單一按鍵的函數 
def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class KeyboardTeleop(Node):
    def __init__(self):
        # 節點名稱
        super().__init__('wasd_keyboard_control')
        # 建立 Publisher，對應原廠底盤控制的 Topic: 'cmd_vel'
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.twist = Twist()
        # 原廠建議的速度設定在 0.35 到 1.0 之間，這裡預設為 0.5
        self.speed = 0.5 

    def run(self, settings):
        print(MSG)
        try:
            while True:
                key = get_key(settings).lower() # 轉小寫，支援大小寫 WASD
                
                if key == 'w':
                    self.twist.linear.x = self.speed
                    self.twist.linear.y = 0.0
                    self.get_logger().info("前進")
                elif key == 's':
                    self.twist.linear.x = -self.speed
                    self.twist.linear.y = 0.0
                    self.get_logger().info("後退")
                elif key == 'a':
                    self.twist.linear.x = 0.0
                    self.twist.linear.y = self.speed  # y軸正值為左平移
                    self.get_logger().info("左平移")
                elif key == 'd':
                    self.twist.linear.x = 0.0
                    self.twist.linear.y = -self.speed # y軸負值為右平移
                    self.get_logger().info("右平移")
                elif key == ' ':
                    self.twist.linear.x = 0.0
                    self.twist.linear.y = 0.0
                    self.get_logger().info("停止")
                else:
                    if (key == '\x03'): # 捕捉 CTRL-C 的中斷訊號
                        break

                # 發布運動指令給底盤
                self.mecanum_pub.publish(self.twist)
                
        except Exception as e:
            print(f"發生錯誤: {e}")
            
        finally:
            # 程式退出時確保車子停止
            self.twist.linear.x = 0.0
            self.twist.linear.y = 0.0
            self.mecanum_pub.publish(self.twist)

def main():
    # 儲存終端機原本的設定
    settings = termios.tcgetattr(sys.stdin)
    
    # 初始化 ROS 2
    rclpy.init()
    node = KeyboardTeleop()
    
    # 建立一個獨立的執行緒來讓 ROS 2 節點持續運作 (spin)，避免卡住按鍵讀取
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,))
    spin_thread.start()

    # 開始執行鍵盤監聽迴圈
    node.run(settings)
    
    # 程式結束清理作業
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()

if __name__ == "__main__":
    main()
