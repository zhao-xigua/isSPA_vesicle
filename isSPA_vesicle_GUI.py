import sys
import numpy as np
import mrcfile
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox, QGridLayout, QSizePolicy
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from generate_shape import generate_ring, generate_elliptical_ring
from image_processing import downsample, center_clip_pad0, center_clip_pad_mean, write_mrc, binary_array
from generate_filled_ellipsoid_mask_no_clip import draw_elliptical_disks
import cv2
import ast
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--bin", type=int, choices=[5, 10], required=True, help="Choose image binning: 5 or 10")

args = parser.parse_args()

DOWNSAMPLE_FACTOR = 2 if args.bin == 5 else 1

class MRCViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("Vesicle Labeling Toolbox")
        self.setGeometry(100, 100, 1500, 1000)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # =========================
        # 顶部按钮区（Import）
        # =========================
        
        top_layout = QHBoxLayout()
        
        top_layout.setContentsMargins(10, 5, 10, 5)
        top_layout.setSpacing(10)
        
        self.import_button = QPushButton("Load MRC")
        self.import_button.setFixedWidth(140)
        self.import_button.clicked.connect(self.import_mrc)
        
        
        self.shape_button = QPushButton("Load Shape")
        self.shape_button.setFixedWidth(140)
        self.shape_button.clicked.connect(self.on_shape_button_click)
        
        top_layout.addWidget(self.import_button)
        top_layout.addWidget(self.shape_button)
        
        self.import_button.setFixedHeight(32)
        self.shape_button.setFixedHeight(32)
        
        top_layout.addStretch()  # 推到左边
        
        self.layout.addLayout(top_layout)
        
        
        # =========================
        # Ellipse 参数区（核心优化）
        # =========================
        
        shape_group = QGroupBox("Ellipse Parameters")
        grid = QGridLayout()
        
        # ---- 控件 ----
        self.axes_input = QLineEdit("(50,50)")
        self.axes_input.returnPressed.connect(self.update_axes)
        
        self.angle_input = QLineEdit("0")
        self.angle_input.returnPressed.connect(self.update_angle)
        
        self.mask_name_input = QLineEdit()
        self.mask_name_input.setPlaceholderText("Enter mask name")
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_data)
        
        # ---- 统一高度 ----
        HEIGHT = 36
        for w in [self.axes_input, self.angle_input, self.mask_name_input]:
            w.setFixedHeight(HEIGHT)
        self.save_button.setFixedHeight(HEIGHT)
        
        # ---- 布局（关键：对齐）----
        grid.addWidget(QLabel("Axes (a, b):"), 0, 0)
        grid.addWidget(self.axes_input, 0, 1)

        grid.addWidget(QLabel("Angle (°):"), 0, 2)
        grid.addWidget(self.angle_input, 0, 3)

        grid.addWidget(QLabel("Mask Name:"), 1, 0)
        grid.addWidget(self.mask_name_input, 1, 1, 1, 3)

        grid.addWidget(self.save_button, 0, 4, 2, 1)
        
        # ---- spacing ----
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(10, 8, 10, 8)

        shape_group.setLayout(grid)
        shape_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(shape_group, stretch=0)
        
        
        # =========================
        # Matplotlib 区
        # =========================

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("border: 1px solid #ccc;")
        self.layout.addWidget(self.canvas, stretch=1)
        
        
        # =========================
        # 全局字体
        # =========================
        
        font = QFont()
        font.setPointSize(11)
        self.setFont(font)
        
        
        # =========================
        # 简单美化（重要）
        # =========================
        
        self.setStyleSheet("""
        QLineEdit {
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 4px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 2px 0 2px;
        }
        """)


        # =========================
        # 原有逻辑（保持不变）
        # =========================
        
        self.filename = ""
        self.processing_image = ""

        self.image_gray = None
        self.axes = (50, 50)
        self.angle = 0

        self.show_ring = True
        self.ring_center = None

        self.ring_list = []
        self._from_mouse_click = False

        self.voxel_size = (1.0, 1.0, 1.0)
        self.ring_artists = []
        
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        
        
    def on_scroll(self, event):
        if self.image_gray is None:
            return
        
        # 鼠标位置（以数据坐标为准）
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        # 当前显示范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 缩放比例（可调）
        base_scale = 1.2

        if event.button == 'up':
            scale_factor = 1 / base_scale   # 放大
        elif event.button == 'down':
            scale_factor = base_scale       # 缩小
        else:
            return

        # 计算新范围（以鼠标为中心缩放）
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        min_size = 20
        max_size = max(self.image_gray.shape)

        if new_width < min_size or new_width > max_size * 5:
            return

        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        new_xmin = xdata - new_width * relx
        new_xmax = xdata + new_width * (1 - relx)
        new_ymin = ydata - new_height * rely
        new_ymax = ydata + new_height * (1 - rely)

        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)

        self.canvas.draw_idle()
    
    
    def redraw_rings(self):
        # 删除旧 ring artist
        for artist in self.ring_artists:
            for coll in artist.collections:
                coll.remove()
                
        self.ring_artists = []

        if self.image_gray is None:
            return

        if not self.show_ring:
            self.canvas.draw_idle()
            return

        for center, axes, angle in self.ring_list:
            ring_mask = generate_elliptical_ring(
                self.image_gray.shape,
                center,
                axes,
                angle,
                thickness=2
            )

            artist = self.ax.contourf(
                ring_mask,
                levels=[0.5, 1],
                colors="green",
                alpha=1
            )

            self.ring_artists.append(artist)

        self.canvas.draw_idle()
    

    def import_mrc(self):
        """ 打开文件对话框选择 MRC 文件，并显示图像 """
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MRC File", "", "MRC Files (*.mrc);;All Files (*)", options=options)
        self.filename = file_path.split("/")[-1]  # 提取文件名（不带路径）

        if file_path:
            self.display_mrc(file_path)

    def display_mrc(self, file_path):
        """ 读取 MRC 并显示 """
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_data = mrc.data.astype(np.float32)
                
        # 归一化到 0-255，转换为 8-bit 灰度图
        image_norm = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
        self.image_gray = image_norm.astype(np.uint8)

        # 更新 Matplotlib 画布
        self.ax.clear()
        self.ax.imshow(self.image_gray, cmap="gray", origin="upper")
        self.ax.set_title(f"MRC Image: {file_path.split('/')[-1]}")
        self.ax.axis("off")
        self.canvas.draw()
        
        self.processing_image = "./image/"+file_path.split("/")[-1].split(".")[0]+".mrc"
        mrc_down = downsample(image_data,DOWNSAMPLE_FACTOR)
        mrc_clip = center_clip_pad_mean(mrc_down)
        write_mrc(self.processing_image, mrc_clip)
        
        
    def import_shape(self):
        """ 调用 generate_elliptical_ring 并绘制 """
        if self.image_gray is not None:
             # 只有鼠标点击时才添加 ring
            if getattr(self, '_from_mouse_click', False):
                self.ring_list.append(((self.image_gray.shape[1] // 2, self.image_gray.shape[0] // 2), self.axes, self.angle))
                self._from_mouse_click = False  # 重置标志

                print(self.ring_list)
                self.ax.clear()
                self.ax.imshow(self.image_gray, cmap="gray", origin="upper")
            
                for center, axes, angle in self.ring_list:
                    if self.show_ring:  # 仅在 self.show_ring 为 True 时绘制圆环
                        ring_mask = generate_elliptical_ring(self.image_gray.shape, center, axes, angle, thickness=2)
                        self.ax.contourf(ring_mask, levels=[0.5, 1], colors="green", alpha=1)
                        
            else:
                self.ax.clear()
                self.ax.imshow(self.image_gray, cmap="gray", origin="upper")
                if self.show_ring and self.ring_center:  # 仅在 self.show_ring 为 True 时绘制圆环
                    ring_mask = generate_elliptical_ring(self.image_gray.shape, self.ring_center, self.axes, self.angle, thickness=2)
                    self.ax.contourf(ring_mask, levels=[0.5, 1], colors="green", alpha=1)  # 填充环形区域
                
            self.ax.set_title(f"Manual Labeling")
            self.ax.axis("off")
            self.canvas.draw()
        else:
            print("请先导入 MRC 图像！")
    
    
    def on_shape_button_click(self):
        self._from_mouse_click = True
        self.import_shape()
    
    def update_list(self):
        """更新 ring_list 中最近一个圆环的中心和半径"""
        if self.ring_list:
            # 更新最后一个圆环的中心和半径
            self.ring_list[-1] = (self.ring_center, self.axes, self.angle)
    
    def update_axes(self):
        """ 更新椭圆长短轴，并重新绘制 """
        try:
            text = self.axes_input.text().strip()
            new_axes = ast.literal_eval(text)
            if (isinstance(new_axes, tuple) and len(new_axes) == 2 and all(isinstance(v, (int, float)) for v in new_axes)):
                
                if new_axes[0] > 0 and new_axes[1] > 0:
                    self.axes = (float(new_axes[0]), float(new_axes[1]))
                    self.update_list()  # 更新列表
                    self.redraw_rings()  # 重新绘制
                    
                else:
                    print("轴长必须大于0!")
                    
            else:
                print("请输入形如 (a, b) 的元组！")
                
        except (ValueError, SyntaxError):
            print("输入格式错误，应为 (a, b)")
            

    def update_angle(self):
        """ 更新圆环半径，并重新绘制 """
        try:
            new_angle = float(self.angle_input.text())  # 读取输入框中的值
            if new_angle > 0:
                self.angle = new_angle
                self.update_list()  # 更新列表
                self.redraw_rings()  # 重新绘制
            else:
                print("请输入大于 0 的 angle！")
        except ValueError:
            print("无效输入，请输入整数！")


            
    def save_data(self):
        """ 保存半径和 mask 以及计算圆心 """
        if self.image_gray is None :
            print("请先导入 MRC 图像并生成形状！")
            return
        
        
        # 获取用户输入的 mask 名称
        mask_name = self.mask_name_input.text().strip()
        if not mask_name:
            print("请输入有效的 mask 名称！")
            return
        
        # 初始化一个全为0的mask图像
        combined_mask = np.zeros(self.image_gray.shape, dtype=np.float32)
        
        if self.ring_center is None:
            combined_mask_binary = combined_mask
        else:
            #for center, axes, angle in self.ring_list:
                #ring_mask = generate_elliptical_ring(self.image_gray.shape, center, axes, angle, thickness=2)
            combined_disk_mask = draw_elliptical_disks(self.ring_list, self.image_gray.shape[1], self.image_gray.shape[0], thick=2)
            #combined_mask = np.logical_or(combined_mask, disk_mask)
            
            #combined_mask = (combined_mask* 255).astype(np.float32)
            #print("combined_mask dtype before saving:", combined_mask.dtype)  
            #print("combined_mask unique values:", np.unique(combined_mask))   # 应为 [  0 255]    
            
            # 处理array成指定格式后输出
            combined_mask_down = downsample(combined_disk_mask,DOWNSAMPLE_FACTOR)
            combined_mask_clip = center_clip_pad0(combined_mask_down)
            combined_mask_binary = binary_array(combined_mask_clip)
        
        
        # 选择保存文件
        save_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
         
        if save_path:
            mask_path = f"{save_path}/{mask_name}.mrc"
            txt_path = f"{save_path}/{mask_name}.txt"
            
            # 保存 mask 二值图像
            #cv2.imwrite(mask_path, (combined_mask * 255).astype(np.uint8))
            
            # 保存为 MRC 格式（注意数据格式和 header）
            with mrcfile.new(mask_path, overwrite=True) as mrc_out:
                mrc_out.set_data(combined_mask_binary)
                mrc_out.voxel_size = self.voxel_size
                
            
            # 保存圆心和半径到文本文件
            with open(txt_path, "w") as f:
            
                for center, axes, angle in self.ring_list:
                
                    a, b = axes
                    axis_ratio = a / (b + 1e-8)
                    shape_type = "circle" if axis_ratio < 1.02 else "ellipse"
                    
                    f.write(
                            f"{self.filename}\t"
                            f"image_size={self.image_gray.shape[1]},{self.image_gray.shape[0]}\t"
                            f"center={center[0]},{center[1]}\t"
                            f"radius={a},{b}\t"
                            f"angle={angle}\t"
                            f"axis_ratio={axis_ratio:.2f}\t"
                            f"type={shape_type}\n"
                            )

            print(f"数据已保存到: {save_path} 和 {txt_path}")
            
            
            
    def on_mouse_press(self, event):
        """ 处理鼠标点击事件 """
        if self.image_gray is None:
            return
        
        if event.button == 2:  # 中键点击
            self.show_ring = not self.show_ring  # 切换是否显示圆环
            self.redraw_rings()  # 重新绘制
        
        if event.button == 1:  # 左键点击，准备拖拽
            if self.show_ring:
                self.ring_center = (int(event.xdata), int(event.ydata))  # 记录鼠标点击位置
                self.update_list()  # 更新列表
                self.redraw_rings()  # 重新绘制

                
                
    def on_mouse_move(self, event):
        """ 处理鼠标拖拽事件 """
        if event.button == 1 and self.show_ring:  # 按住左键并移动
            if event.xdata is not None and event.ydata is not None:
                self.ring_center = (int(event.xdata), int(event.ydata))  # 更新圆环中心
                self.update_list()  # 更新列表
                #print(self.ring_center)
                self.redraw_rings()  # 实时更新绘制

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MRCViewer()
    window.show()
    sys.exit(app.exec())
