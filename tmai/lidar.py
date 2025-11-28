import cv2
import numpy as np
import time
import win32gui
import win32con
import win32ui
import ctypes
from ctypes import windll
import math

class TrackmaniaLidar:
    def __init__(self):
        self.window_width = 1080
        self.window_height = 1080
        self.fps = 30
        
        # Ray configuration
        self.main_ray_count = 11
        self.tire_ray_count = 8 
        self.ray_count = self.main_ray_count + self.tire_ray_count
        self.rays = []
        self.ray_colors = []
        self.ray_lengths = []
        self.ray_angles = []
        

        self.black_threshold = 55
        
        # Max distance corresponds to ~50m in Trackmania
        self.max_ray_length = 1000 
        self.meter_to_pixel_ratio = 20  
        
        # Distance threshold for color change (green to red) in meters
        self.danger_distance_threshold = 1.0
        

        self.perspective_strength = 0.5 
        self.central_ray_boost = 1.2  
        self.depth_factor = 0.5      
        
  
        self.safe_color = (0, 255, 0) 
        self.danger_color = (0, 0, 255) 
        self.tire_ray_color = (0, 255, 255)
        
 
        for i in range(self.main_ray_count):
            self.ray_colors.append(self.safe_color)
        

        for i in range(self.tire_ray_count):
            self.ray_colors.append(self.tire_ray_color)
        
        self.trackmania_hwnd = None
        self.running = False
    
    def find_trackmania_window(self):
        """Find the TrackMania window handle"""
        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and "Trackmania" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)
            return True
        
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        
        if not hwnds:
            print("TrackMania window not found.")
            return None
        
        return hwnds[0]
    
    def resize_trackmania_window(self):
        """Resize the TrackMania window to 1080x1080"""
        if self.trackmania_hwnd:

            x, y, right, bottom = win32gui.GetWindowRect(self.trackmania_hwnd)
            

            user32 = ctypes.windll.user32
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            
            new_x = 10
            new_y = 10
            

            win32gui.SetWindowPos(
                self.trackmania_hwnd,
                win32con.HWND_TOP,
                new_x, new_y,
                self.window_width, self.window_height,
                win32con.SWP_SHOWWINDOW
            )
            
            # Try to set foreground window, but don't fail if it doesn't work
            try:
                win32gui.SetForegroundWindow(self.trackmania_hwnd)
            except Exception as e:
                print(f"⚠ Could not set TrackMania window to foreground: {e}")
            
            print(f"TrackMania window resized to {self.window_width}x{self.window_height}")
            return True
        return False
    
    def capture_screen(self):
        """Capture the TrackMania window content"""
        if not self.trackmania_hwnd:
            return None
        

        left, top, right, bottom = win32gui.GetWindowRect(self.trackmania_hwnd)
        width = 1330
        height = 1080
        

        hwnd_dc = win32gui.GetWindowDC(self.trackmania_hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        

        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        

        result = windll.user32.PrintWindow(self.trackmania_hwnd, save_dc.GetSafeHdc(), 3)
        

        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        

        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.trackmania_hwnd, hwnd_dc)
        
        return img
    
    def cast_rays(self, frame):
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        

        center_x = w // 2
        center_y = int(620) 
        

        angle_range = 160 
        main_angles = np.linspace(-angle_range/2, angle_range/2, self.main_ray_count)
        

        left_tire_x = center_x - 400
        right_tire_x = center_x + 400
        tire_y = center_y + 20 
        

        left_tire_angles = [-90, -70] 
        right_tire_angles = [70, 90]  
        
      
        self.ray_angles = list(main_angles) + left_tire_angles + right_tire_angles
        

        self.rays = []
        self.ray_lengths = []
        

        for i, angle in enumerate(main_angles):
 
            rad_angle = math.radians(angle)
            

            dx = math.sin(rad_angle)
            dy = -math.cos(rad_angle) 
            

            position ="center"
        
            ray_length, ray_length_meters = self.cast_single_ray(frame, center_x, center_y, dx, dy, position, angle)
            

            end_x = int(center_x + dx * ray_length)
            end_y = int(center_y + dy * ray_length)
            

            if ray_length_meters < self.danger_distance_threshold:
                ray_color = self.danger_color  # Red if close to obstacle
            else:
                ray_color = self.safe_color  # Green if safe distance
                

            self.ray_colors[i] = ray_color
            

            cv2.line(result_frame, (center_x, center_y), (end_x, end_y), ray_color, 2)
            

            self.rays.append((end_x, end_y))
 

            text_pos = ((center_x + end_x) // 2, (center_y + end_y) // 2)
            cv2.putText(result_frame, f"{ray_length_meters:.1f}m | {angle:.0f}°", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ray_color, 1)
        

        for i, angle in enumerate(left_tire_angles):
            idx = self.main_ray_count + i
            rad_angle = math.radians(angle)
            dx = math.sin(rad_angle)
            dy = -math.cos(rad_angle)

            ray_length, ray_length_meters = self.cast_single_ray(frame, left_tire_x, tire_y, dx, dy, "left", angle)
            
            end_x = int(left_tire_x + dx * ray_length)
            end_y = int(tire_y + dy * ray_length)
            

            if ray_length_meters < self.danger_distance_threshold:
                ray_color = self.danger_color 
            else:
                ray_color = self.tire_ray_color 
                

            self.ray_colors[idx] = ray_color
            
            cv2.line(result_frame, (left_tire_x, tire_y), (end_x, end_y), ray_color, 2)
            self.rays.append((end_x, end_y))
 
            
            text_pos = ((left_tire_x + end_x) // 2, (tire_y + end_y) // 2)
            cv2.putText(result_frame, f"{ray_length_meters:.1f}m | {angle:.0f}°", text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, ray_color, 1)
            

        for i, angle in enumerate(right_tire_angles):
            idx = self.main_ray_count + len(left_tire_angles) + i
            rad_angle = math.radians(angle)
            dx = math.sin(rad_angle)
            dy = -math.cos(rad_angle)
            
            ray_length, ray_length_meters = self.cast_single_ray(frame, right_tire_x, tire_y, dx, dy, "right", angle)

            
            end_x = int(right_tire_x + dx * ray_length)
            end_y = int(tire_y + dy * ray_length)
            

            if ray_length_meters < self.danger_distance_threshold:
                ray_color = self.danger_color 
            else:
                ray_color = self.tire_ray_color 
                

            self.ray_colors[idx] = ray_color
            
            cv2.line(result_frame, (right_tire_x, tire_y), (end_x, end_y), ray_color, 2)
            self.rays.append((end_x, end_y))

            
            text_pos = ((right_tire_x + end_x) // 2, (tire_y + end_y) // 2)
            cv2.putText(result_frame, f"{ray_length_meters:.1f}m | {angle:.0f}°", text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, ray_color, 1)

        cv2.circle(result_frame, (center_x, center_y), 5, (0, 255, 255), -1) 
        cv2.circle(result_frame, (left_tire_x, tire_y), 5, (0, 255, 255), -1) 
        cv2.circle(result_frame, (right_tire_x, tire_y), 5, (0, 255, 255), -1)  
        
 
        self.ray_lengths = []
        

        for i in range(len(self.rays)):
    
            end_point = self.rays[i]
            
  
            if i < self.main_ray_count:
                ray_type = "Front"
                angle = self.ray_angles[i]
                start_point = (center_x, center_y)
            elif i < self.main_ray_count + len(left_tire_angles):
                ray_type = "L Tire"
                angle = self.ray_angles[i]
                start_point = (left_tire_x, tire_y)
            else:
                ray_type = "R Tire"
                angle = self.ray_angles[i]
                start_point = (right_tire_x, tire_y)
            

            pixel_length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            

            abs_angle_from_center = abs(angle)
            

            base_distance = pixel_length / self.meter_to_pixel_ratio
            

            if abs_angle_from_center >= 89.0:
                cos_factor = 0.1  
            else:
                cos_factor = math.cos(math.radians(abs_angle_from_center))
                

            perspective_factor = 1.0
            central_angle_threshold = 50
            is_mid_angle = 30 <= abs_angle_from_center <= 60
            
            if abs_angle_from_center < central_angle_threshold:
                center_weight = 1.0 - (abs_angle_from_center / central_angle_threshold)
                
                if is_mid_angle:
                    mid_angle_factor = 1.0 - abs(abs_angle_from_center - 45) / 15
                    center_weight = center_weight * (1.0 + mid_angle_factor * 0.5)
                    
                depth_effect = 1.0 + (pixel_length / 300) * self.depth_factor
                central_boost = 1.0 + (self.central_ray_boost - 1.0) * center_weight
                perspective_factor = depth_effect * central_boost * (1.0 + self.perspective_strength * center_weight)
            else:
                side_weight = (abs_angle_from_center - central_angle_threshold) / (90 - central_angle_threshold)
                perspective_factor = 1.0 + (1.0 - side_weight) * 0.3 * self.perspective_strength
                

            adjusted_distance = base_distance * cos_factor * perspective_factor
            
  
            if pixel_length > 300:
                camera_height_factor = 1.0 + (pixel_length - 300) / 700.0 * self.perspective_strength
                adjusted_distance *= camera_height_factor
            

            text = f"{ray_type} {i}: {adjusted_distance:.1f}m ({angle:.0f}°)"
            cv2.putText(result_frame, text, (10, 30 + 20*i), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ray_colors[i], 1)
        
        return result_frame
        
    def cast_single_ray(self, frame, start_x, start_y, dx, dy, position, angle):
        """Helper method to cast a single ray and find obstacle distance with perspective correction"""
        h, w = frame.shape[:2]
        max_len = self.max_ray_length
        ray_length = max_len
        

        normalized_angle = angle
        

        abs_angle_from_center = abs(normalized_angle)
        

        for length in range(10, max_len, 2):
            x = int(start_x + dx * length)
            y = int(start_y + dy * length)

            if x < 0 or y < 0 or x >= w or y >= h:
                ray_length = length
                break

            pixel = frame[y, x]
            

            if (pixel[0] < self.black_threshold and 
                pixel[1] < self.black_threshold and 
                pixel[2] < self.black_threshold):
                ray_length = length
                break
                

        base_distance = ray_length / self.meter_to_pixel_ratio
        

        angle_rad = math.radians(abs_angle_from_center)
        if abs_angle_from_center >= 89.0: 
            cos_factor = 0.1  
        else:
            cos_factor = math.cos(angle_rad) 

        perspective_factor = 1.0
        central_angle_threshold = 50  
        mid_angle_range = (30, 60)    
        

        is_mid_angle = mid_angle_range[0] <= abs_angle_from_center <= mid_angle_range[1]
        
        if abs_angle_from_center < central_angle_threshold:

            center_weight = 1.0 - (abs_angle_from_center / central_angle_threshold)
            

            if is_mid_angle:
   
                mid_angle_factor = 1.0 - abs(abs_angle_from_center - 45) / 15  
                center_weight = center_weight * (1.0 + mid_angle_factor * 0.5)  
            
            depth_effect = 1.0 + (ray_length / 300) * self.depth_factor
            

            central_boost = 1.0 + (self.central_ray_boost - 1.0) * center_weight
            
 
            perspective_factor = depth_effect * central_boost * (1.0 + self.perspective_strength * center_weight)
        else:

            side_weight = (abs_angle_from_center - central_angle_threshold) / (90 - central_angle_threshold)
            perspective_factor = 1.0 + (1.0 - side_weight) * 0.3 * self.perspective_strength
        

        ray_distance = base_distance * cos_factor * perspective_factor

        if ray_length > 300:
            camera_height_factor = 1.0 + (ray_length - 300) / 700.0 * self.perspective_strength
            ray_distance *= camera_height_factor
            
        return ray_length, ray_distance
    
    def run(self):
        """Main loop for the TrackMania LIDAR system"""

        self.trackmania_hwnd = self.find_trackmania_window()
        if not self.trackmania_hwnd:
            print("TrackMania not running. Please start the game first.")
            return
        
        self.resize_trackmania_window()
        time.sleep(1) 
        

        cv2.namedWindow("TrackMania LIDAR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TrackMania LIDAR", 1080, 1080)
        

        frame_time = 1/self.fps
        prev_time = time.time()
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        self.running = True
        print("LIDAR system running. Press ESC to exit.")
        
        while self.running:
     
            current_time = time.time()
            elapsed = current_time - prev_time
            
            if elapsed < frame_time:
                time.sleep(max(0, frame_time - elapsed))
                current_time = time.time()
                elapsed = current_time - prev_time
            
            prev_time = current_time
            
  
            fps_counter += 1
            if current_time - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = current_time
            

            frame = self.capture_screen()
            if frame is None:
                print("Failed to capture screen. Retrying...")
                time.sleep(0.5)
                self.trackmania_hwnd = self.find_trackmania_window()
                continue
            

            processed_frame = self.cast_rays(frame)
            

            cv2.putText(
                processed_frame, 
                f"FPS: {current_fps}", 
                (processed_frame.shape[1] - 140, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            

            cv2.putText(
                processed_frame, 
                f"Danger Threshold: {self.danger_distance_threshold:.1f}m", 
                (processed_frame.shape[1] - 250, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255),  # Red color for danger threshold
                1
            )
            
            cv2.putText(
                processed_frame, 
                f"Pixel-to-Meter: {self.meter_to_pixel_ratio:.1f}", 
                (processed_frame.shape[1] - 220, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            

            cv2.putText(
                processed_frame, 
                f"Perspective: {self.perspective_strength:.1f} | Center Boost: {self.central_ray_boost:.1f} | Depth: {self.depth_factor:.1f}", 
                (10, processed_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 0), 
                1
            )
            
            # Show result
            cv2.imshow("TrackMania LIDAR", processed_frame)
            
   
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  
                self.running = False
            elif key == ord('+'): 
                self.meter_to_pixel_ratio += 1
                print(f"Meter-to-pixel ratio: {self.meter_to_pixel_ratio}")
            elif key == ord('-'): 
                self.meter_to_pixel_ratio = max(1, self.meter_to_pixel_ratio - 1)
                print(f"Meter-to-pixel ratio: {self.meter_to_pixel_ratio}")
            elif key == ord('d'): 
                self.danger_distance_threshold += 1.0
                print(f"Danger threshold: {self.danger_distance_threshold}m")
            elif key == ord('a'): 
                self.danger_distance_threshold = max(1.0, self.danger_distance_threshold - 1.0)
                print(f"Danger threshold: {self.danger_distance_threshold}m")
            elif key == ord('p'):  
                self.perspective_strength = min(1.0, self.perspective_strength + 0.1)
                print(f"Perspective strength: {self.perspective_strength:.1f}")
            elif key == ord('o'): 
                self.perspective_strength = max(0.0, self.perspective_strength - 0.1)
                print(f"Perspective strength: {self.perspective_strength:.1f}")
            elif key == ord('c'): 
                self.central_ray_boost += 0.1
                print(f"Central ray boost: {self.central_ray_boost:.1f}")
            elif key == ord('x'): 
                self.central_ray_boost = max(1.0, self.central_ray_boost - 0.1)
                print(f"Central ray boost: {self.central_ray_boost:.1f}")
            elif key == ord('f'): 
                self.depth_factor += 0.1
                print(f"Depth factor: {self.depth_factor:.1f}")
            elif key == ord('v'): 
                self.depth_factor = max(0.1, self.depth_factor - 0.1)
                print(f"Depth factor: {self.depth_factor:.1f}")
        
  
        cv2.destroyAllWindows()
        print("LIDAR system stopped.")

if __name__ == "__main__":
    print("Starting TrackMania LIDAR visualization system...")
    lidar = TrackmaniaLidar()
    lidar.run()