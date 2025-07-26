import sys, os, cv2, re, pyqtgraph, datetime, heapq
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from csaps import csaps
from tkinter import filedialog
from tqdm import trange
from tifffile import imread,TiffWriter
from PyQt5 import Qt, QtCore, QtGui
from PyQt5.QtCore import QPoint, QThread
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow
from GUI import Ui_MainWindow
from registration.default_ops import reg_default_ops
from regismodule import reg
from denoisemodule import deep_denoise
from denoisemodule_srdtrans import srd_denoise
from segmodule import seg, cal_diam
from cellpose.default_ops import seg_default_ops
from Cascade.Demo_scripts.Demo_predict import cas
from Cascade.cascade2p.utils_discrete_spikes import infer_discrete_spikes
from skimage import measure
from scipy.io import loadmat, savemat
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'black')


class NIPWindow(QMainWindow, Ui_MainWindow):
    #label9 is the label widget to show raw Image
    def __init__(self, parent=None):
        super(NIPWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)  # 去掉标题栏的代码
        self.setWindowFlags(Qt.Qt.FramelessWindowHint)
        # self.pushButton_resize.clicked.connect(self.resize_win)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.lineEdit_smooth.setValidator(QtGui.QDoubleValidator(0.0, 50000.0, 3))
        self.lineEdit_maxregshift.setValidator(QtGui.QDoubleValidator(0.0, 50000.0, 3))
        self.lineEdit_smoothtime.setValidator(QtGui.QDoubleValidator(0.0, 50000.0, 3))
        self.lineEdit_nimg_init.setValidator(QtGui.QIntValidator(0, 50000))
        self.lineEdit_diameter.setValidator(QtGui.QDoubleValidator(0.0, 50000.0, 3))
        self.lineEdit_diameter.textChanged.connect(self.diameter_change)
        self.lineEdit_diameter_2.textChanged.connect(self.diameter_change_2)
        self.lineEdit_flowthr.setValidator(QtGui.QDoubleValidator(0.0, 50000.0, 3))
        self.lineEdit_flowthr.textChanged.connect(self.flow_change)
        self.lineEdit_flowthr_2.textChanged.connect(self.flow_change_2)
        self.lineEdit_cellprobthr.setValidator(QtGui.QDoubleValidator(-6, 50000.0, 3))
        self.lineEdit_cellprobthr.textChanged.connect(self.cellprob_change)
        self.lineEdit_cellprobthr_2.textChanged.connect(self.cellprob_change_2)
        self.horizontalScrollBar_Raw.valueChanged.connect(self.slice_show)
        self.horizontalScrollBar_Enhance.valueChanged.connect(self.slice_show)
        self.comboBox_featuremap.currentIndexChanged.connect(self.feature_change)
        self.checkBox_seg_part.stateChanged.connect(self.seg_part_change)
        self.checkBox_seg_part_2.stateChanged.connect(self.seg_part_change_2)
        self.checkBox_maskfix.stateChanged.connect(self.maskfix_change)
        self.checkBox_maskfix_2.stateChanged.connect(self.maskfix_change_2)

        self.Slider_Contrast.setValue(100)
        self.Slider_Contrast_2.setValue(100)
        self.Slider_Brightness.setValue(550)
        self.Slider_Brightness_2.setValue(200)
        self.Slider_Contrast.valueChanged.connect(self.contrast_adj)
        self.Slider_Contrast.valueChanged.connect(self.slice_show)
        self.Slider_Contrast_2.valueChanged.connect(self.slice_show)
        self.Slider_Brightness.valueChanged.connect(self.brightness_adj)
        self.Slider_Brightness.valueChanged.connect(self.slice_show)
        self.Slider_Brightness_2.valueChanged.connect(self.slice_show)
        self.pushButton_load_tiff.clicked.connect(self.load_tiff)
        self.pushButton_load_masks.clicked.connect(self.load_masks)
        self.pushButton_load_masks_2.clicked.connect(self.load_masks)
        self.pushButton_load_result.clicked.connect(self.load_result)
        self.pushButton_load_result_2.clicked.connect(self.load_result)
        self.pushButton_save_tiff.clicked.connect(self.savetiff)
        self.pushButton_load_enhanced.clicked.connect(self.load_enhanced)
        self.pushButton_run_enhance.clicked.connect(self.run_enhance)
        self.pushButton_cal_diameter.clicked.connect(self.cal_diameter)
        self.pushButton_cal_diameter_2.clicked.connect(self.cal_diameter)
        self.pushButton_run_seg.clicked.connect(self.seg_ops)
        self.pushButton_run_extract.clicked.connect(self.run_extract)
        self.lineEdit_8.editingFinished.connect(self.choose_signal_show)
        self.pushButton_run_discrete.clicked.connect(self.run_discrete)
        self.pushButton_run_spike.clicked.connect(self.run_cas)
        self.pushButton_run_all.clicked.connect(self.run_all)
        self.pushButton_save_seg.clicked.connect(self.save_masks)
        self.pushButton_save_sig.clicked.connect(self.save_raw_sig)
        self.pushButton_save_spike.clicked.connect(self.save_spk)
        self.pushButton_save_discrete.clicked.connect(self.save_discrete)
        self.pushButton_save_all.clicked.connect(self.save_all)
        self.pushButton_run_batches.clicked.connect(self.run_batches)

        self.top_flag = False
        self.l9_flag = False
        self.l10_flag = False
        self.l16_flag = False
        self.llargeseg_flag = False
        self.all_flag = False

        self.contrast = 1
        self.brightness = 5.5
        self.colorma = plt.get_cmap('Paired', 12)
        self.batches_pipe = 0

        self.scale_ratio = 0.1
        self.lastpoint = QPoint()
        self.nextpoint = QPoint()
        self.roi_flag = False
        self.ctrl_flag = False
        self.point = []  #to save the roi coordinate that is manually selected
        self.labelsize_9 = [self.label_9.x(),self.label_9.y(),self.label_9.width(),self.label_9.height()]
        self.labelsize_10 = [self.label_10.x(), self.label_10.y(), self.label_10.width(), self.label_10.height()]
        self.labelsize_16 = [self.label_16.x(), self.label_16.y(), self.label_16.width(), self.label_16.height()]
        self.labelsize_largeseg = [self.label_largeseg.x(), self.label_largeseg.y(), self.label_largeseg.width(), self.label_largeseg.height()]
        self.deep_model_path = './DeepCAD_RT/pth'
        self.deep_model_name = os.listdir(self.deep_model_path)
        self.srd_model_path = './SRDTrans/pth'
        self.srd_model_name = os.listdir(self.srd_model_path)
        self.comboBox_denoisemodel.clear()
        self.comboBox_denoisemodel.addItems(self.deep_model_name)
        self.comboBox_denoisealgo.currentIndexChanged.connect(self.denoise_algorithm_change)

        self.seg_model_path = './cellpose/model'
        self.seg_model_name = os.listdir(self.seg_model_path)
        self.comboBox_segmodel.clear()
        self.comboBox_segmodel.addItems(self.seg_model_name)
        self.comboBox_segmodel.setCurrentText('cyto')

        self.cas_model_path = './Cascade/Pretrained_models'
        self.cas_model_name = os.listdir(self.cas_model_path)
        self.cas_model_name = [item for item in self.cas_model_name if ".yaml" not in item]
        self.comboBox_spikemodel.clear()
        self.comboBox_spikemodel.addItems(self.cas_model_name)

    def flow_change(self):
        self.lineEdit_flowthr_2.setText(self.lineEdit_flowthr.text())

    def flow_change_2(self):
        self.lineEdit_flowthr.setText(self.lineEdit_flowthr_2.text())

    def cellprob_change(self):
        self.lineEdit_cellprobthr_2.setText(self.lineEdit_cellprobthr.text())

    def cellprob_change_2(self):
        self.lineEdit_cellprobthr.setText(self.lineEdit_cellprobthr_2.text())

    def diameter_change(self):
        self.lineEdit_diameter_2.setText(self.lineEdit_diameter.text())

    def diameter_change_2(self):
        self.lineEdit_diameter.setText(self.lineEdit_diameter_2.text())

    def seg_part_change(self):
        self.checkBox_seg_part_2.setChecked(self.checkBox_seg_part.isChecked())

    def seg_part_change_2(self):
        self.checkBox_seg_part.setChecked(self.checkBox_seg_part_2.isChecked())

    def maskfix_change(self):
        self.checkBox_maskfix_2.setChecked(self.checkBox_maskfix.isChecked())

    def maskfix_change_2(self):
        self.checkBox_maskfix.setChecked(self.checkBox_maskfix_2.isChecked())

    def denoise_algorithm_change(self):
        self.comboBox_denoisemodel.clear()
        Algorithm = self.comboBox_denoisealgo.currentText()
        if Algorithm  == 'SRDTrans':
            self.comboBox_denoisemodel.addItems(self.srd_model_name)
        elif Algorithm  == 'DeepCAD_RT':
            self.comboBox_denoisemodel.addItems(self.deep_model_name)
        else:
            print("Invalid Algorithm")

    def mousePressEvent(self, event):
        self.m_win_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
        self.m_l9_Position = self.label_9.mapFromGlobal(Qt.QCursor.pos())
        self.m_l10_Position = self.label_10.mapFromGlobal(Qt.QCursor.pos())
        self.m_l16_Position = self.label_16.mapFromGlobal(Qt.QCursor.pos())
        self.m_llargeseg_Position = self.label_largeseg.mapFromGlobal(Qt.QCursor.pos())
        self.m_f41_Position = self.frame_41.mapFromGlobal(Qt.QCursor.pos())
        self.m_f42_Position = self.frame_42.mapFromGlobal(Qt.QCursor.pos())
        self.m_f45_Position = self.frame_45.mapFromGlobal(Qt.QCursor.pos())
        self.m_f15_Position = self.frame_15.mapFromGlobal(Qt.QCursor.pos())
        xtop = self.frame_5.x()
        ytop = self.frame_5.y()
        heitop = self.frame_5.height()
        widtop = self.frame_5.width()
        heif41 = self.frame_41.height()
        widf41 = self.frame_41.width()
        heif42 = self.frame_42.height()
        widf42 = self.frame_42.width()
        heif45 = self.frame_45.height()
        widf45 = self.frame_45.width()
        heif15 = self.frame_15.height()
        widf15 = self.frame_15.width()
        if self.m_win_Position.x()>xtop and self.m_win_Position.x()<xtop+widtop and \
                self.m_win_Position.y()>ytop and self.m_win_Position.y()<ytop+heitop:
        # 1/3 mouse events function to move the window or image
            if event.button() == Qt.Qt.LeftButton and self.isMaximized() == False:
                self.top_flag = True
                event.accept()
                self.setCursor(Qt.QCursor(Qt.Qt.ClosedHandCursor))  # 更改鼠标图标
        elif self.tabWidget.currentIndex()==0 and self.m_f41_Position.x()>0 and self.m_f41_Position.x()<widf41 and \
                self.m_f41_Position.y()>0 and self.m_f41_Position.y()<heif41:
            if event.button() == Qt.Qt.LeftButton:
                self.l9_flag = True
                event.accept()
                self.setCursor(Qt.QCursor(Qt.Qt.ClosedHandCursor))
        elif self.tabWidget.currentIndex()==0 and self.m_f42_Position.x()>0 and self.m_f42_Position.x()<widf42 and \
                self.m_f42_Position.y()>0 and self.m_f42_Position.y()<heif42:
            if event.button() == Qt.Qt.LeftButton:
                self.l10_flag = True
                event.accept()
                self.setCursor(Qt.QCursor(Qt.Qt.ClosedHandCursor))
        elif self.tabWidget.currentIndex()==1 and self.m_f45_Position.x()>0 and self.m_f45_Position.x()<widf45 and \
                self.m_f45_Position.y()>0 and self.m_f45_Position.y()<heif45:
            # The action trigger of manually select neurons
            if event.button() == Qt.Qt.LeftButton:
                self.l16_flag = True
                event.accept()
                self.setCursor(Qt.QCursor(Qt.Qt.ClosedHandCursor))
            if event.button() == Qt.Qt.RightButton:
                if hasattr(myWin,'pix16'):
                    self.frame.setMouseTracking(True)
                    self.frame_mid.setMouseTracking(True)
                    self.tabWidget.setMouseTracking(True)
                    self.centralwidget.setMouseTracking(True)
                    self.label_16.setMouseTracking(True)
                    self.frame_45.setMouseTracking(True)
                    self.frame_30.setMouseTracking(True)
                    self.frame_23.setMouseTracking(True)
                    self.tab_2.setMouseTracking(True)
                    self.tab_2.parent().setMouseTracking(True)
                    self.setMouseTracking(True)
                    self.outflag = False
                    self.roi_flag = True
                    self.point = []
                    self.startpoint = (event.globalPos() - self.label_16.mapToGlobal(
                        QPoint(0, 0))) * self.Lx / self.label_16.width()
                    self.lastpoint = self.startpoint
                    self.nextpoint = self.lastpoint
        elif self.tabWidget.currentIndex()==2 and self.m_f15_Position.x()>0 and self.m_f15_Position.x()<widf15 and \
                self.m_f15_Position.y()>0 and self.m_f15_Position.y()<heif15:
            # The action trigger of manually select neurons
            if event.button() == Qt.Qt.LeftButton:
                self.llargeseg_flag = True
                event.accept()
                self.setCursor(Qt.QCursor(Qt.Qt.ClosedHandCursor))
            if event.button() == Qt.Qt.RightButton:
                if hasattr(myWin,'pix16'):
                    self.frame.setMouseTracking(True)
                    self.frame_mid.setMouseTracking(True)
                    self.tabWidget.setMouseTracking(True)
                    self.centralwidget.setMouseTracking(True)
                    self.label_largeseg.setMouseTracking(True)
                    self.frame_15.setMouseTracking(True)
                    self.frame_72.setMouseTracking(True)
                    self.tab_3.setMouseTracking(True)
                    self.tab_3.parent().setMouseTracking(True)
                    self.setMouseTracking(True)
                    self.outflag = False
                    self.roi_flag = True
                    self.point = []
                    self.startpoint = (self.label_largeseg.mapFromGlobal(Qt.QCursor.pos())) * self.Lx / self.label_largeseg.width()
                    self.lastpoint = self.startpoint
                    self.nextpoint = self.lastpoint


    def mouseMoveEvent(self, event):
        # 2/3 mouse events function to move the window or image
        if Qt.Qt.LeftButton and self.top_flag:
            self.move(event.globalPos() - self.m_win_Position)  # 更改窗口位置
            event.accept()
        elif Qt.Qt.LeftButton and self.l9_flag:
            self.label_9.move(event.globalPos() - self.frame_41.mapToGlobal(Qt.QPoint(0,0)) - self.m_l9_Position)
            self.labelsize_9 = [self.label_9.x(), self.label_9.y(), self.label_9.width(), self.label_9.height()]
            event.accept()
        elif Qt.Qt.LeftButton and self.l10_flag:
            self.label_10.move(event.globalPos() - self.frame_42.mapToGlobal(Qt.QPoint(0,0)) - self.m_l10_Position)
            self.labelsize_10 = [self.label_10.x(), self.label_10.y(), self.label_10.width(), self.label_10.height()]
            event.accept()
        elif Qt.Qt.LeftButton and self.l16_flag:
            self.label_16.move(event.globalPos() - self.frame_45.mapToGlobal(Qt.QPoint(0,0)) - self.m_l16_Position)
            self.labelsize_16 = [self.label_16.x(), self.label_16.y(), self.label_16.width(), self.label_16.height()]
            event.accept()
        elif Qt.Qt.LeftButton and self.llargeseg_flag:
            self.label_largeseg.move(self.frame_15.mapFromGlobal(Qt.QCursor.pos()) - self.m_llargeseg_Position)
            self.labelsize_largeseg = [self.label_largeseg.x(), self.label_largeseg.y(), self.label_largeseg.width(), self.label_largeseg.height()]
            event.accept()
        elif self.roi_flag is True:
            # The action of manually select neurons
            if self.tabWidget.currentIndex() == 1 :
                self.nextpoint = (event.globalPos() - self.label_16.mapToGlobal(QPoint(0, 0)))*self.Lx/self.label_16.width()
            elif self.tabWidget.currentIndex() == 2 :
                self.nextpoint = (self.label_largeseg.mapFromGlobal(Qt.QCursor.pos()))* self.Lx / self.label_largeseg.width()
            if abs((self.nextpoint.x() - self.startpoint.x()) **2 + (
                    self.nextpoint.y() - self.startpoint.y()) **2) > 10:
                self.outflag = True
            if self.outflag and abs((self.nextpoint.x() - self.startpoint.x()) **2 + (
                    self.nextpoint.y() - self.startpoint.y()) **2) < 10:
                self.lastpoint = QPoint()
                self.nextpoint = QPoint()
                self.roi_flag = False
                mask = np.zeros([self.Ly, self.Lx])
                pts = []
                for i in range(len(self.point)):
                    pts.append([self.point[i].x(), self.point[i].y()])
                mask = cv2.fillPoly(np.uint8(mask), [np.int32(pts)], 1)
                property_temp = measure.regionprops(np.uint(mask))[0]
                one_property = {'area': property_temp.area,
                                'centroid': np.array(property_temp.centroid),
                                'coords': property_temp.coords,
                                'color': self.colorma.colors[self.n_roi % 11,:]}
                self.properties.append(one_property)
                self.n_roi += 1
                temp_mask = cv2.Canny(np.uint8(mask), 0, 0) == 255
                self.rainbowedge[0, temp_mask] += np.uint8(255 * one_property['color'][0])
                self.rainbowedge[1, temp_mask] += np.uint8(255 * one_property['color'][1])
                self.rainbowedge[2, temp_mask] += np.uint8(255 * one_property['color'][2])
                # coloredge[0, :, :] = cv2.Canny(np.uint8(self.masks[-1, :, :]), 0, 0) * self.colorma.colors[i % 11, 0]
                # coloredge[1, :, :] = cv2.Canny(np.uint8(self.masks[-1, :, :]), 0, 0) * self.colorma.colors[i % 11, 1]
                # coloredge[2, :, :] = cv2.Canny(np.uint8(self.masks[-1, :, :]), 0, 0) * self.colorma.colors[i % 11, 2]
                # self.rainbowedge = self.rainbowedge + coloredge
                self.rainbowedge = np.uint8(self.rainbowedge)
                self.binaedge = np.asarray(
                    Image.fromarray(np.uint8(self.rainbowedge.transpose(1, 2, 0)), 'RGB').convert('L'))
                self.binaedge[self.binaedge == 0] = 1
                self.binaedge[self.binaedge > 1] = 0
                rainbowmask = self.binaedge * self.feature_image_enhance + self.rainbowedge
                rainbowmask = rainbowmask.transpose(1, 2, 0)
                rainbowmask_pil = Image.fromarray(rainbowmask, 'RGB')
                # add index text
                ft = ImageFont.truetype(font='./Font/msyhbd.ttc', size=12)

                img_Q = QtGui.QImage(rainbowmask_pil.tobytes(), rainbowmask.shape[1], rainbowmask.shape[0],
                                     rainbowmask.shape[1] * 3, QtGui.QImage.Format_RGB888)
                self.pix16 = QtGui.QPixmap(img_Q)
                self.label_16.setPixmap(self.pix16)
                self.label_16.setScaledContents(True)
                self.label_16.setGeometry(
                    QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2],
                                 self.labelsize_16[3]))
                self.label_largeseg.setPixmap(self.pix16)
                self.label_largeseg.setScaledContents(True)
                self.label_largeseg.setGeometry(
                    QtCore.QRect(self.labelsize_largeseg[0], self.labelsize_largeseg[1], self.labelsize_largeseg[2],
                                 self.labelsize_largeseg[3]))
                # self.mask_show()
                self.update()
                self.label_n_roi.setText(str(self.n_roi))
                self.label_n_roi_2.setText(str(self.n_roi))
                self.frame.setMouseTracking(False)
                self.frame_mid.setMouseTracking(False)
                self.tabWidget.setMouseTracking(False)
                self.centralwidget.setMouseTracking(False)
                self.label_16.setMouseTracking(False)
                self.frame_45.setMouseTracking(False)
                self.frame_30.setMouseTracking(False)
                self.frame_23.setMouseTracking(False)
                self.tab_2.setMouseTracking(False)
                self.tab_2.parent().setMouseTracking(False)
                self.frame_15.setMouseTracking(False)
                self.frame_72.setMouseTracking(False)
                self.tab_3.setMouseTracking(False)
                self.tab_3.parent().setMouseTracking(False)
                self.setMouseTracking(False)
                return
            self.point.append(self.nextpoint)
            self.update()  # 更新绘图事件,每次执行update都会触发一次paintEvent(self, event)函数
            self.label_16.setPixmap(self.pix16)
            self.label_16.setScaledContents(True)
            self.label_largeseg.setPixmap(self.pix16)
            self.label_largeseg.setScaledContents(True)

    def mouseReleaseEvent(self, event):
        # 3/3 mouse events function to move the window or image
        self.top_flag = False
        self.l9_flag = False
        self.l10_flag = False
        self.l16_flag = False
        self.llargeseg_flag = False
        self.setCursor(Qt.QCursor(Qt.Qt.ArrowCursor))

    def paintEvent(self, event):  # 重写paintEvent事件
        if hasattr(myWin, 'pix16'):
            pp = QPainter(self.pix16)
            pen = QPen(QtCore.Qt.green, 1, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            pp.setPen(pen)
            pp.drawLine(self.lastpoint, self.nextpoint)
            self.lastpoint = self.nextpoint


    def wheelEvent(self, event):
        self.m_f41_Position = self.frame_41.mapFromGlobal(Qt.QCursor.pos()) # 获取鼠标相对控件的位置
        self.m_f42_Position = self.frame_42.mapFromGlobal(Qt.QCursor.pos())
        self.m_f45_Position = self.frame_45.mapFromGlobal(Qt.QCursor.pos())
        self.m_f15_Position = self.frame_15.mapFromGlobal(Qt.QCursor.pos())
        coordinate_f41 = [self.m_f41_Position.x(),self.m_f41_Position.y()] #相对坐标
        coordinate_f42 = [self.m_f42_Position.x(), self.m_f42_Position.y()]
        coordinate_f45 = [self.m_f45_Position.x(), self.m_f45_Position.y()]
        coordinate_f15 = [self.m_f15_Position.x(), self.m_f15_Position.y()]
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        heif41 = self.frame_41.height()
        widf41 = self.frame_41.width()
        heif42 = self.frame_42.height()
        widf42 = self.frame_42.width()
        heif45 = self.frame_45.height()
        widf45 = self.frame_45.width()
        heif15 = self.frame_15.height()
        widf15 = self.frame_15.width()
        if self.tabWidget.currentIndex()==0 and coordinate_f41[0] > 0 and coordinate_f41[0] < widf41 and coordinate_f41[1] > 0 and coordinate_f41[1] <heif41:
            if angleY > 0:
                self.scale_img(2,coordinate_f41,'label_9')
            else:  # 滚轮下滚
                self.scale_img(-2,coordinate_f41,'label_9')
        elif self.tabWidget.currentIndex()==0 and coordinate_f42[0] > 0 and coordinate_f42[0] < widf42 and coordinate_f42[1] > 0 and coordinate_f42[1] <heif42:
            if angleY > 0:
                self.scale_img(2,coordinate_f42,'label_10')
            else:  # 滚轮下滚
                self.scale_img(-2,coordinate_f42,'label_10')
        elif self.tabWidget.currentIndex() == 1 and coordinate_f45[0] > 0 and coordinate_f45[0] < widf45 and \
                coordinate_f45[1] > 0 and coordinate_f45[1] < heif45:
            if angleY > 0:
                self.scale_img(2, coordinate_f45, 'label_16')
            else:  # 滚轮下滚
                self.scale_img(-2, coordinate_f45, 'label_16')
        elif self.tabWidget.currentIndex() == 2 and coordinate_f15[0] > 0 and coordinate_f15[0] < widf15 and \
                coordinate_f15[1] > 0 and coordinate_f15[1] < heif15:
            if angleY > 0:
                self.scale_img(2, coordinate_f15, 'label_largeseg')
            else:  # 滚轮下滚
                self.scale_img(-2, coordinate_f15, 'label_largeseg')

    def keyPressEvent(self, event):
        #to delete the roi choosed
        self.m_f45_Position =  self.frame_45.mapFromGlobal(Qt.QCursor.pos())
        coordinate_f45 = [self.m_f45_Position.x(), self.m_f45_Position.y()]
        self.m_f15_Position = self.frame_15.mapFromGlobal(Qt.QCursor.pos())
        coordinate_f15 = [self.m_f15_Position.x(), self.m_f15_Position.y()]
        if hasattr(myWin, 'properties') and (self.tabWidget.currentIndex() == 1 and coordinate_f45[0] > 0 and \
                coordinate_f45[0] < self.frame_45.width() and coordinate_f45[1] > 0 and coordinate_f45[1] < self.frame_45.height())\
                or (self.tabWidget.currentIndex() == 2 and coordinate_f15[0] > 0 and \
                coordinate_f15[0] < self.frame_15.width() and coordinate_f15[1] > 0 and coordinate_f15[1] < self.frame_15.height()):
            if self.tabWidget.currentIndex() == 1:
                which_roi = None
                m_l16_Position = (self.label_16.mapFromGlobal(Qt.QCursor.pos())) * self.Lx / self.label_16.width()
                for i in range(self.n_roi):
                    judge_array = np.int8(self.properties[i]['coords'] == [m_l16_Position.y(), m_l16_Position.x()])
                    judge_line = judge_array[:, 0] + judge_array[:, 1]
                    if 2 in judge_line:
                        which_roi = i
                move_step = 10
                multi_move = ([self.frame_45.width()][0] / self.labelsize_16[3]) ** 2
                move_step = round(move_step * multi_move)
            elif self.tabWidget.currentIndex() == 2:
                which_roi = None
                m_llargeseg_Position = (self.label_largeseg.mapFromGlobal(Qt.QCursor.pos())) * self.Lx / self.label_largeseg.width()
                for i in range(self.n_roi):
                    judge_array = np.int8(self.properties[i]['coords'] == [m_llargeseg_Position.y(), m_llargeseg_Position.x()])
                    judge_line = judge_array[:,0]+judge_array[:,1]
                    if 2 in judge_line:
                        which_roi = i
                move_step = 10
                multi_move = ([self.frame_15.width()][0] / self.labelsize_largeseg[3]) ** 2
                move_step = round(move_step * multi_move)
            if event.key() == QtCore.Qt.Key_E and which_roi is not None:
                    key = 'E'
                    self.n_roi -= 1
                    if which_roi != 0:
                        temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                        temp_mask[self.properties[which_roi]['coords'][:, 0],
                                  self.properties[which_roi]['coords'][:, 1]] = 1
                        temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                        self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][0])
                        self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][1])
                        self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][2])
                        self.properties = self.properties[:which_roi] + self.properties[which_roi+1:]
                        self.mask_change()
                        self.label_n_roi.setText(str(self.n_roi))
                        self.label_n_roi_2.setText(str(self.n_roi))
                    else:
                        temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                        temp_mask[self.properties[which_roi]['coords'][:, 0],
                                  self.properties[which_roi]['coords'][:, 1]] = 1
                        self.properties = self.properties[which_roi + 1:]
                        self.mask_change()
                        self.label_n_roi.setText(str(self.n_roi))
                        self.label_n_roi_2.setText(str(self.n_roi))

            if event.key() == QtCore.Qt.Key_W:
                key = 'W'
                if move_step < 1:
                    move_step = 1
                if which_roi is None:
                    for i in range(3):
                        self.rainbowedge[i,:] = np.roll(self.rainbowedge[i,:], -move_step, 0)
                    for i in range(self.n_roi):
                        self.properties[i]['coords'][:, 0] -= move_step
                        self.properties[i]['centroid'][0] -= move_step
                        if ~(self.properties[i]['coords'][:, 0] > -1).all():
                            self.properties[i]['coords'][:, 0] += move_step
                            self.properties[i]['centroid'][0] += move_step
                            temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                            temp_mask[self.properties[i]['coords'][:, 0],
                                      self.properties[i]['coords'][:, 1]] = 1
                            temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                            self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[i]['color'][2])
                            temp_mask = np.roll(temp_mask, -move_step, 0)
                            self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[i]['color'][2])
                else:
                    temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                    temp_mask[self.properties[which_roi]['coords'][:, 0],
                              self.properties[which_roi]['coords'][:, 1]] = 1
                    temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                    self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][2])
                    self.properties[which_roi]['coords'][:, 0] -= move_step
                    self.properties[which_roi]['centroid'][0] -= move_step
                    temp_mask = np.roll(temp_mask, -move_step, 0)
                    if ~(self.properties[which_roi]['coords'][:, 0] > -1).all():
                        self.properties[which_roi]['coords'][:, 0] += move_step
                        self.properties[which_roi]['centroid'][0] += move_step
                        temp_mask = np.roll(temp_mask, +move_step, 0)
                    self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][2])
                self.mask_change()

            if event.key() == QtCore.Qt.Key_S:
                key = 'S'
                if move_step < 1:
                    move_step = 1
                if which_roi is None:
                    for i in range(3):
                        self.rainbowedge[i, :] = np.roll(self.rainbowedge[i, :], move_step, 0)
                    for i in range(self.n_roi):
                        self.properties[i]['coords'][:, 0] += move_step
                        self.properties[i]['centroid'][0] += move_step
                        if ~(self.properties[i]['coords'][:, 0] < self.Ly).all():
                            self.properties[i]['coords'][:, 0] -= move_step
                            self.properties[i]['centroid'][0] -= move_step
                            temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                            temp_mask[self.properties[i]['coords'][:, 0],
                                      self.properties[i]['coords'][:, 1]] = 1
                            temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                            self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[i]['color'][2])
                            temp_mask = np.roll(temp_mask, +move_step, 0)
                            self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[i]['color'][2])
                else:
                    temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                    temp_mask[self.properties[which_roi]['coords'][:, 0],
                              self.properties[which_roi]['coords'][:, 1]] = 1
                    temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                    self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][2])
                    self.properties[which_roi]['coords'][:, 0] += move_step
                    self.properties[which_roi]['centroid'][0] += move_step
                    temp_mask = np.roll(temp_mask, move_step, 0)
                    if ~(self.properties[which_roi]['coords'][:, 0] < self.Ly).all():
                        self.properties[which_roi]['coords'][:, 0] -= move_step
                        self.properties[which_roi]['centroid'][0] -= move_step
                        temp_mask = np.roll(temp_mask, -move_step, 0)
                    self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][2])
                self.mask_change()

            if event.key() == QtCore.Qt.Key_A:
                key = 'A'
                if move_step < 1:
                    move_step = 1
                if which_roi is None:
                    for i in range(3):
                        self.rainbowedge[i, :] = np.roll(self.rainbowedge[i, :], -move_step, 1)
                    for i in range(self.n_roi):
                        self.properties[i]['coords'][:, 1] -= move_step
                        self.properties[i]['centroid'][1] -= move_step
                        if ~(self.properties[i]['coords'][:, 1] > -1).all():
                            self.properties[i]['coords'][:, 1] += move_step
                            self.properties[i]['centroid'][1] += move_step
                            temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                            temp_mask[self.properties[i]['coords'][:, 0],
                                      self.properties[i]['coords'][:, 1]] = 1
                            temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                            self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[i]['color'][2])
                            temp_mask = np.roll(temp_mask, -move_step, 1)
                            self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[i]['color'][2])
                else:
                    temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                    temp_mask[self.properties[which_roi]['coords'][:, 0],
                              self.properties[which_roi]['coords'][:, 1]] = 1
                    temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                    self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][2])
                    self.properties[which_roi]['coords'][:, 1] -= move_step
                    self.properties[which_roi]['centroid'][1] -= move_step
                    temp_mask = np.roll(temp_mask, -move_step, 1)
                    if ~(self.properties[which_roi]['coords'][:, 1] > -1).all():
                        self.properties[which_roi]['coords'][:, 1] += move_step
                        self.properties[which_roi]['centroid'][1] += move_step
                        temp_mask = np.roll(temp_mask, +move_step, 1)
                    self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][2])
                self.mask_change()

            if event.key() == QtCore.Qt.Key_D:
                key = 'D'
                if move_step < 1:
                    move_step = 1
                if which_roi is None:
                    for i in range(3):
                        self.rainbowedge[i, :] = np.roll(self.rainbowedge[i, :], move_step, 1)
                    for i in range(self.n_roi):
                        self.properties[i]['coords'][:, 1] += move_step
                        self.properties[i]['centroid'][1] += move_step
                        if ~(self.properties[i]['coords'][:, 1] < self.Lx).all():
                            self.properties[i]['coords'][:, 1] -= move_step
                            self.properties[i]['centroid'][1] -= move_step
                            temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                            temp_mask[self.properties[i]['coords'][:, 0],
                                      self.properties[i]['coords'][:, 1]] = 1
                            temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                            self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[i]['color'][2])
                            temp_mask = np.roll(temp_mask, move_step, 1)
                            self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[i]['color'][0])
                            self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[i]['color'][1])
                            self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[i]['color'][2])
                else:
                    temp_mask = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
                    temp_mask[self.properties[which_roi]['coords'][:, 0],
                              self.properties[which_roi]['coords'][:, 1]] = 1
                    temp_mask = cv2.Canny(np.uint8(temp_mask), 0, 0) == 255
                    self.rainbowedge[0, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] -= np.uint8(255 * self.properties[which_roi]['color'][2])
                    self.properties[which_roi]['coords'][:, 1] += move_step
                    self.properties[which_roi]['centroid'][1] += move_step
                    temp_mask = np.roll(temp_mask, move_step, 1)
                    if ~(self.properties[which_roi]['coords'][:, 1] < self.Lx).all():
                        self.properties[which_roi]['coords'][:, 1] -= move_step
                        self.properties[which_roi]['centroid'][1] -= move_step
                        temp_mask = np.roll(temp_mask, -move_step, 1)
                    self.rainbowedge[0, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][0])
                    self.rainbowedge[1, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][1])
                    self.rainbowedge[2, temp_mask] += np.uint8(255 * self.properties[which_roi]['color'][2])
                self.mask_change()

    def mask_change(self):
        self.rainbowedge = np.uint8(self.rainbowedge)
        self.binaedge = np.asarray(
            Image.fromarray(np.uint8(self.rainbowedge.transpose(1, 2, 0)), 'RGB').convert('L'))
        self.binaedge[self.binaedge == 0] = 1
        self.binaedge[self.binaedge > 1] = 0
        rainbowmask = self.binaedge * self.feature_image_enhance + self.rainbowedge
        rainbowmask = rainbowmask.transpose(1, 2, 0)
        rainbowmask_pil = Image.fromarray(rainbowmask, 'RGB')
        # add index text
        ft = ImageFont.truetype(font='./Font/msyhbd.ttc', size=12)

        img_Q = QtGui.QImage(rainbowmask_pil.tobytes(), rainbowmask.shape[1], rainbowmask.shape[0],
                             rainbowmask.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.pix16 = QtGui.QPixmap(img_Q)
        self.label_16.setPixmap(self.pix16)
        self.label_16.setScaledContents(True)
        self.label_16.setGeometry(
            QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2],
                         self.labelsize_16[3]))
        self.label_largeseg.setPixmap(self.pix16)
        self.label_largeseg.setScaledContents(True)
        self.label_largeseg.setGeometry(
            QtCore.QRect(self.labelsize_largeseg[0], self.labelsize_largeseg[1], self.labelsize_largeseg[2],
                         self.labelsize_largeseg[3]))
        # self.mask_show()
        self.update()

    #窗口大小更改
    def resize_win(self):
        if  self.isMaximized():
            self.showNormal()
            self.pushButton_resize.setIcon(Qt.QIcon("./icon/24gl-fullScreenEnter3 (1).png"))
        else:
            self.showMaximized()
            self.pushButton_resize.setIcon(Qt.QIcon("./icon/24gl-fullScreenEnter3 (1).png"))
        self.labelsize_9 = [0, 0, self.frame_41.width(), self.frame_41.height()]
        self.label_9.setGeometry(
            QtCore.QRect(self.labelsize_9[0], self.labelsize_9[1], self.labelsize_9[2], self.labelsize_9[3]))
        self.labelsize_10 = [0, 0, self.frame_42.width(), self.frame_42.height()]
        self.label_10.setGeometry(
            QtCore.QRect(self.labelsize_10[0], self.labelsize_10[1], self.labelsize_10[2], self.labelsize_10[3]))
        self.labelsize_16 = [0, 0, self.frame_45.width(), self.frame_45.height()]
        self.label_16.setGeometry(
            QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2], self.labelsize_16[3]))
        self.labelsize_17 = [0, 0, self.frame_46.width(), self.frame_46.height()]
        self.label_17.setGeometry(
            QtCore.QRect(self.labelsize_17[0], self.labelsize_17[1], self.labelsize_17[2], self.labelsize_17[3]))
        self.labelsize_18 = [0, 0, self.frame_47.width(), self.frame_47.height()]
        self.label_18.setGeometry(
            QtCore.QRect(self.labelsize_18[0], self.labelsize_18[1], self.labelsize_18[2], self.labelsize_18[3]))
        self.labelsize_19 = [0, 0, self.frame_48.width(), self.frame_48.height()]
        self.label_19.setGeometry(
            QtCore.QRect(self.labelsize_19[0], self.labelsize_19[1], self.labelsize_19[2], self.labelsize_19[3]))

    def load_tiff(self):
        # This function loads the selected tiff file and display
        self._translate = QtCore.QCoreApplication.translate
        root = tk.Tk()
        root.withdraw()
        # get path
        self.load_path = filedialog.askopenfilename(filetypes=[("tif文件", ".tif")])
        (self.file_name, ext) = os.path.splitext(os.path.basename(self.load_path))
        if self.load_path == '':    # return when you did not choose a tiff file(That's when it would return a empty string to load_path)
            return
        self.clearGUI()
        # raw_image preprocess
        self.raw_image = imread(self.load_path)
        #self.raw_image=self.raw_image[:,:,0:274]
        self.raw_image[self.raw_image < 0] = 0
        hist = np.histogram(self.raw_image[0,:,:], bins=256)
        if hist[1][np.where(hist[0] == np.max(hist[0]))[0]] >= 30000:
            self.raw_image = self.raw_image - 32768
            self.raw_image[self.raw_image > 32768] = 0
        if len(self.raw_image.shape) == 3:
            self.n_frames, self.Ly, self.Lx = self.raw_image.shape
        elif len(self.raw_image.shape) == 4:
            self.n_frames = self.raw_image.shape[0]
            self.Ly = self.raw_image.shape[2]
            self.Lx = self.raw_image.shape[3]
        else:
            print("Image properties error")
            return
        self.raw_image_show = self.bitconvert(self.raw_image)
        self.label_26.setText(self._translate("MainWindow", "1/{}".format(self.n_frames)))
        self.horizontalScrollBar_Raw.setMaximum(self.n_frames)
        first=self.raw_image_show[0,:,:]
        pix = self.con_imgtopixmap(first)
        self.label_9.setPixmap(pix)
        self.label_9.setScaledContents(True)
        self.labelsize_9 = [0, 0, self.frame_41.width(), self.frame_41.height()]
        self.label_9.setGeometry(
            QtCore.QRect(self.labelsize_9[0], self.labelsize_9[1], self.labelsize_9[2], self.labelsize_9[3]))
        print('The file ' + self.load_path + ' loads successfully')

    def load_enhanced(self):
        self._translate = QtCore.QCoreApplication.translate
        root = tk.Tk()
        root.withdraw()
        # get path
        enhanced_path = filedialog.askopenfilename(filetypes=[("tif文件", ".tif")])
        (self.file_name, ext) = os.path.splitext(os.path.basename(enhanced_path))
        if enhanced_path == '':    # return when you did not choose a tiff file(That's when it would return a empty string to load_path)
            return
        self.clearGUI()
        self.enhanced_image = imread(enhanced_path)
        self.n_frames, self.Ly, self.Lx = self.enhanced_image.shape
        hist = np.histogram(self.enhanced_image[0, :, :], bins=256)
        if hist[1][np.where(hist[0] == np.max(hist[0]))[0]] >= 30000:
            self.enhanced_image = self.enhanced_image - 32768
            self.enhanced_image[self.enhanced_image > 32768] = 0
        self.enhanced_image_show = self.bitconvert(self.enhanced_image)
        # self.feature_image = np.mean(self.enhanced_image[120:135,:,:], axis=0)
        self.std_image = np.std(self.enhanced_image, axis=0)
        self.mean_image = np.mean(self.enhanced_image, axis=0)
        self.max_image = np.max(self.enhanced_image, axis=0)
        self.featuremap = self.comboBox_featuremap.currentText()
        if self.featuremap == 'Standard deviation' and hasattr(myWin, 'std_image'):
            self.feature_image = self.std_image
        elif self.featuremap == 'Mean' and hasattr(myWin, 'mean_image'):
            self.feature_image = self.mean_image
        elif self.featuremap == 'Max' and hasattr(myWin, 'max_image'):
            self.feature_image = self.max_image
        self.horizontalScrollBar_Enhance.setMaximum(self.n_frames)
        self.label_27.setText(self._translate("MainWindow", "1/{}".format(self.n_frames)))
        self.labelsize_10 = [0, 0, self.frame_42.width(), self.frame_42.height()]
        self.label_10.setGeometry(
            QtCore.QRect(self.labelsize_10[0], self.labelsize_10[1], self.labelsize_10[2], self.labelsize_10[3]))
        self.slice_show()
        print('Enhanced image loaded')

    def load_masks(self):
        root = tk.Tk()
        root.withdraw()
        # get path
        masks_path = filedialog.askopenfilename(filetypes=[("mat文件", ".mat")])
        if masks_path == '':
            return
        try:
            self.masks = loadmat(masks_path)['masks'].transpose(2,0,1)
            self.n_roi = np.shape(self.masks)[0]
            properties = list(range(self.n_roi))
            for i in range(self.n_roi):
                properties[i] = measure.regionprops(np.uint8(self.masks[i, :, :]))[0]
            if self.n_roi != 0:
                self.properties = list()
            for i in range(self.n_roi):
                one_property = {'area': properties[i].area,
                                'centroid': np.array(properties[i].centroid),
                                'coords': properties[i].coords,
                                'color': self.colorma.colors[i % 11, :]}
                self.properties.append(one_property)
            self.label_n_roi.setText(str(self.n_roi))
            self.label_n_roi_2.setText(str(self.n_roi))
            print("Masks loaded")
        except:
            try:
                properties = loadmat(masks_path)['properties'][0]
                self.properties = list()
                self.n_roi = len(properties)
                for i in range(self.n_roi):
                    one_property = {'area': np.squeeze(properties[i]['area'][0][0]),
                                    'centroid': np.array(properties[i]['centroid'][0][0][0]),
                                    'coords': properties[i]['coords'][0][0],
                                    'color': properties[i]['color'][0][0][0]}
                    self.properties.append(one_property)
                self.label_n_roi.setText(str(self.n_roi))
                self.label_n_roi_2.setText(str(self.n_roi))
                print('Properties loaded')
            except:
                print("Masks load error")
        if hasattr(myWin, 'feature_image'):
            self.mask_show()

    def load_result(self):
        root = tk.Tk()
        root.withdraw()
        # get path
        result_path = filedialog.askopenfilename(filetypes=[("mat文件", ".mat")])
        if result_path == '':
            return
        (self.file_name, ext) = os.path.splitext(os.path.basename(result_path))
        self.clearGUI()
        try:
            self.feature_image = loadmat(result_path)['feature_image']
        except:
            None
        try:
            self.masks = loadmat(result_path)['masks']
        except:
            None
        try:
            properties = loadmat(result_path)['properties'][0]
            self.properties = list()
            self.n_roi = len(properties)
            for i in range(self.n_roi):
                one_property = {'area': np.squeeze(properties[i]['area'][0][0]),
                                'centroid': np.array(properties[i]['centroid'][0][0][0]),
                                'coords': properties[i]['coords'][0][0],
                                'color': properties[i]['color'][0][0][0]}
                self.properties.append(one_property)
        except:
            None
        try:
            self.raw_sig = loadmat(result_path)['raw_sig']
        except:
            None
        try:
            self.dff_sig = loadmat(result_path)['dff_sig']
        except:
            None
        try:
            self.spk_sig = loadmat(result_path)['spk_sig']
        except:
            None
        try:
            self.discrete_sig = loadmat(result_path)['discrete_sig']
        except:
            None
        finally:
            if hasattr(myWin, 'feature_image') and hasattr(myWin, 'masks'):
                self.masks = self.masks.transpose(2,0,1)
                self.Ly = self.masks.shape[1]
                self.Lx = self.masks.shape[2]
                self.n_roi = self.masks.shape[0]
                properties = list(range(self.n_roi))
                for i in range(self.n_roi):
                    properties[i] = measure.regionprops(np.uint8(self.masks[i, :, :]))[0]
                if self.n_roi != 0:
                    self.properties = list()
                for i in range(self.n_roi):
                    one_property = {'area': properties[i].area,
                                    'centroid': np.array(properties[i].centroid),
                                    'coords': properties[i].coords,
                                    'color': self.colorma.colors[i % 11, :]}
                    self.properties.append(one_property)
                print(self.masks.shape)
                self.mask_show()
            if hasattr(myWin, 'feature_image') and hasattr(myWin, 'properties'):
                self.Ly = self.feature_image.shape[0]
                self.Lx = self.feature_image.shape[1]
                self.mask_show()
            if hasattr(myWin, 'dff_sig'):
                self.n_roi = self.dff_sig.shape[0]
                self.n_frames = self.dff_sig.shape[1]
                if self.n_roi > 10:
                    self.show_dff(range(1, 11))
                else:
                    self.show_dff(range(self.n_roi))
            if hasattr(myWin, 'spk_sig'):
                self.n_roi = self.spk_sig.shape[0]
                self.n_frames = self.spk_sig.shape[1]
                if self.n_roi > 10:
                    self.show_spk(range(1, 11))
                else:
                    self.show_spk(range(self.n_roi))
            if hasattr(myWin, 'discrete_sig'):
                img_pil = Image.fromarray(np.uint8(self.discrete_sig))
                self.pix19 = img_pil.toqpixmap()
                self.label_19.setPixmap(self.pix19)
                self.label_19.setScaledContents(True)
            self.label_n_roi.setText(str(self.n_roi))
            self.label_n_roi_2.setText(str(self.n_roi))
            print('Result :' + self.file_name +' loaded')
            return

    def savetiff(self):
        # save the enhanced tiff if exist
        if hasattr(myWin,'enhanced_image'):
            root = tk.Tk()
            root.withdraw()
            self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name, filetypes=[("tif文件", ".tif")])
            if self.load_path == '' or self.save_path == '':
                return
            with TiffWriter(self.save_path + '.tif', bigtiff=True) as tif:
                tif.write(np.fix(self.enhanced_image).astype(np.int16))
            print("Save enhanced image finished")
        # else:
        #     print('There is no enhanced image')


    def slice_show(self):
        if (self.horizontalScrollBar_Raw.valueChanged or self.Slider_Contrast.valueChanged) and hasattr(myWin,'raw_image'):
            slice = self.horizontalScrollBar_Raw.sliderPosition()
            pix = self.con_imgtopixmap(self.raw_image_show[slice-1,:,:])
            self.label_9.setPixmap(pix)
            self.label_9.setScaledContents(True)
            self.label_9.setGeometry(
                QtCore.QRect(self.labelsize_9[0], self.labelsize_9[1], self.labelsize_9[2], self.labelsize_9[3]))
            self.label_26.setText(self._translate("MainWindow", "{}/{}".format(slice,self.n_frames)))
        if (self.horizontalScrollBar_Enhance.valueChanged or self.Slider_Contrast.valueChanged) and \
                hasattr(myWin, 'enhanced_image') and self.enhanced_image is not None:
            slice = self.horizontalScrollBar_Enhance.sliderPosition()
            pix = self.con_imgtopixmap(self.enhanced_image_show[slice - 1, :, :])
            self.label_10.setPixmap(pix)
            self.label_10.setScaledContents(True)
            self.label_10.setGeometry(
                QtCore.QRect(self.labelsize_10[0], self.labelsize_10[1], self.labelsize_10[2], self.labelsize_10[3]))
            self.label_27.setText(self._translate("MainWindow", "{}/{}".format(slice, self.n_frames)))
        if  (self.Slider_Contrast_2.valueChanged or self.Slider_Brightness_2.valueChanged) and hasattr(myWin,'rainbowedge')\
                and hasattr(myWin, 'feature_image'):
            img_pil = Image.fromarray(self.bitconvert(self.feature_image), 'L')
            img_pil_enhance = ImageEnhance.Contrast(img_pil).enhance(self.Slider_Contrast_2.value() / 100)
            img_pil_enhance = ImageEnhance.Brightness(img_pil_enhance).enhance(self.Slider_Brightness_2.value() / 100)
            self.feature_image_enhance = np.asarray(img_pil_enhance)
            rainbowmask = self.binaedge * self.feature_image_enhance + self.rainbowedge
            rainbowmask = rainbowmask.transpose(1, 2, 0)
            rainbowmask_pil = Image.fromarray(rainbowmask, 'RGB')
            # add index text
            ft = ImageFont.truetype(font='./Font/msyhbd.ttc', size=12)

            img_Q = QtGui.QImage(rainbowmask_pil.tobytes(), rainbowmask.shape[1], rainbowmask.shape[0],
                                 rainbowmask.shape[1] * 3, QtGui.QImage.Format_RGB888)
            self.pix16 = QtGui.QPixmap(img_Q)
            self.label_16.setPixmap(self.pix16)
            self.label_16.setScaledContents(True)
            self.label_16.setGeometry(
                QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2], self.labelsize_16[3]))
            self.label_largeseg.setPixmap(self.pix16)
            self.label_largeseg.setScaledContents(True)
            self.label_largeseg.setGeometry(
                QtCore.QRect(self.labelsize_largeseg[0], self.labelsize_largeseg[1], self.labelsize_largeseg[2], self.labelsize_largeseg[3]))

    def feature_change(self):
        self.featuremap = self.comboBox_featuremap.currentText()
        if self.featuremap == 'Standard deviation' and hasattr(myWin, 'std_image'):
            self.feature_image = self.std_image
        elif self.featuremap == 'Mean' and hasattr(myWin, 'mean_image'):
            self.feature_image = self.mean_image
        elif self.featuremap == 'Max' and hasattr(myWin, 'max_image'):
            self.feature_image = self.max_image
        if hasattr(myWin,'properties'):
            self.mask_show()

    def con_imgtopixmap(self,img):
        """
        convert image to pix map for show

        param img: image for convert

        return pix: pixmap for show
        """
        img_pil = Image.fromarray(img)
        con_img = ImageEnhance.Contrast(img_pil).enhance(self.contrast)
        con_img = ImageEnhance.Brightness(con_img).enhance(self.brightness)
        pix = con_img.toqpixmap()
        return pix

    def contrast_adj(self):
        self.contrast = self.Slider_Contrast.value()/100


    def brightness_adj(self):
        self.brightness = self.Slider_Brightness.value()/100


    def bitconvert(self,img):
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
        img = np.uint8(img)
        return img

    def scale_img(self,flag,coordinate,Widgetname):
        """
        Zoom in or out of the displayed image

        param flag: Flag used to determine whether to zoom in or out of the image
        param coordinate: The coordinates of the mouse relative to the Widget
        param Widgetname: Which image to rescale

        """

        if Widgetname=='label_9':
             xl9 = self.label_9.x()
             yl9 = self.label_9.y()
             widl9 = self.label_9.width()
             heil9 = self.label_9.height()
             xratio = (coordinate[0]-xl9)/widl9
             yratio = (coordinate[1]-yl9)/heil9
             self.labelsize_9 = [
                 int(xl9-widl9*(flag*self.scale_ratio)*xratio),
                 int(yl9-heil9*(flag*self.scale_ratio)*yratio),
                 int(widl9*(1+flag*self.scale_ratio)),
                 int(heil9*(1+flag*self.scale_ratio))]
             self.label_9.setGeometry(QtCore.QRect(self.labelsize_9[0],self.labelsize_9[1],self.labelsize_9[2],self.labelsize_9[3]))
        elif Widgetname=='label_10':
             xl10 = self.label_10.x()
             yl10 = self.label_10.y()
             widl10 = self.label_10.width()
             heil10 = self.label_10.height()
             xratio = (coordinate[0] - xl10) / widl10
             yratio = (coordinate[1] - yl10) / heil10
             self.labelsize_10 = [
                 int(xl10 - widl10 * (flag * self.scale_ratio) * xratio),
                 int(yl10 - heil10 * (flag * self.scale_ratio) * yratio),
                 int(widl10 * (1 + flag * self.scale_ratio)),
                 int(heil10 * (1 + flag * self.scale_ratio))]
             self.label_10.setGeometry(
                 QtCore.QRect(self.labelsize_10[0], self.labelsize_10[1], self.labelsize_10[2], self.labelsize_10[3]))
        elif Widgetname == 'label_16':
            xl16 = self.label_16.x()
            yl16 = self.label_16.y()
            widl16 = self.label_16.width()
            heil16 = self.label_16.height()
            xratio = (coordinate[0] - xl16) / widl16
            yratio = (coordinate[1] - yl16) / heil16
            self.labelsize_16 = [
                int(xl16 - widl16 * (flag * self.scale_ratio) * xratio),
                int(yl16 - heil16 * (flag * self.scale_ratio) * yratio),
                int(widl16 * (1 + flag * self.scale_ratio)),
                int(heil16 * (1 + flag * self.scale_ratio))]
            self.label_16.setGeometry(
                QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2], self.labelsize_16[3]))
        elif Widgetname == 'label_largeseg':
            xllargeseg = self.label_largeseg.x()
            yllargeseg = self.label_largeseg.y()
            widllargeseg = self.label_largeseg.width()
            heillargeseg = self.label_largeseg.height()
            xratio = (coordinate[0] - xllargeseg) / widllargeseg
            yratio = (coordinate[1] - yllargeseg) / heillargeseg
            self.labelsize_largeseg = [
                int(xllargeseg - widllargeseg * (flag * self.scale_ratio) * xratio),
                int(yllargeseg - heillargeseg * (flag * self.scale_ratio) * yratio),
                int(widllargeseg * (1 + flag * self.scale_ratio)),
                int(heillargeseg * (1 + flag * self.scale_ratio))]
            self.label_largeseg.setGeometry(
                QtCore.QRect(self.labelsize_largeseg[0], self.labelsize_largeseg[1], self.labelsize_largeseg[2], self.labelsize_largeseg[3]))
        # else if Widgetname == 'label_16':
        # else if Widgetname == 'label_17':
        # else if Widgetname == 'label_18':
        # else if Widgetname == 'label_19':

    def run_enhance(self):
        # run Image enhance,which includes registration and denoise
        algorithm = self.comboBox_denoisealgo.currentText()
        model = self.comboBox_denoisemodel.currentText()
        if self.lineEdit_smooth.text() != '' and self.lineEdit_maxregshift.text() != '' and self.lineEdit_smoothtime.text() != '' and \
                self.lineEdit_nimg_init.text() != '' and hasattr(myWin,'raw_image'):
            reg_ops = reg_default_ops()
            reg_ops['smooth_sigma'] = float(self.lineEdit_smooth.text())
            reg_ops['maxregshift'] = float(self.lineEdit_maxregshift.text())
            reg_ops['smooth_sigma_time'] = float(self.lineEdit_smoothtime.text())
            reg_ops['nimg_init'] = int(self.lineEdit_nimg_init.text())
            if self.checkBox_nonrigid.isChecked():
                reg_ops['nonrigid'] = True
                print('----------- Nonrigid')
            if self.checkBox_bidiphase.isChecked():
                reg_ops['do_bidiphase'] = 1
                reg_ops['bidi_corrected'] = 1
            self.enhanceTh = enhanceThread(self.raw_image, reg_ops, algorithm, model)
            self.enhanceTh.start()
            self.enhanceTh.finished.connect(self.Thread_post_process)
            self.enhanceTh.finished.connect(self.slice_show)
            # self.enhanced_image_show = self.bitconvert(self.enhanced_image)
            # self.horizontalScrollBar_Enhance.setMaximum(self.n_frames)
            # first = self.enhanced_image_show[0, :, :]
            # pix = self.con_imgtopixmap(first)
            # self.label_10.setPixmap(pix)
            # self.label_10.setScaledContents(True)
            # self.labelsize_10 = [0, 0, self.frame_42.width(), self.frame_42.height()]
            # self.label_10.setGeometry(
            #     QtCore.QRect(self.labelsize_10[0], self.labelsize_10[1], self.labelsize_10[2], self.labelsize_10[3]))
            # # Obtain an std map for subsequent cell segmentation

        else:
            return

    def cal_diameter(self):
        if hasattr(myWin,'feature_image'):
            seg_ops = seg_default_ops()
            seg_ops['model_type'] = self.seg_model_path + '\\' + self.comboBox_segmodel.currentText()
            seg_ops['pretrained_size'] = './cellpose/size_model' + '\\'\
                                         'size_%s%s_0.npy' % ('cyto', 'torch')
            diameter = cal_diam(self.feature_image,seg_ops)
            self.lineEdit_diameter.setText(str(diameter))
            self.lineEdit_diameter_2.setText(str(diameter))
        else:
            return
    def seg_ops(self):
        if self.checkBox_seg_part.isChecked():
            self.seg_part()
        else:
            self.run_seg()

    def run_seg(self):
        if hasattr(myWin,'enhanced_image') and self.isfloat(self.lineEdit_flowthr.text()) and \
                self.isfloat(self.lineEdit_cellprobthr.text()) and self.isfloat(self.lineEdit_diameter.text())\
                and (not self.checkBox_maskfix.isChecked() or not hasattr(myWin,'masks')):
            seg_ops = seg_default_ops()
            seg_ops['flow_threshold'] = float(self.lineEdit_flowthr.text())
            seg_ops['cellprob_threshold'] = float(self.lineEdit_cellprobthr.text())
            seg_ops['diameter'] = float(self.lineEdit_diameter.text())
            seg_ops['model_type'] = self.seg_model_path + '\\' + self.comboBox_segmodel.currentText()
            self.featuremap = self.comboBox_featuremap.currentText()
            if self.featuremap == 'Standard deviation' and hasattr(myWin, 'std_image'):
                self.feature_image = self.std_image
            elif self.featuremap == 'Mean' and hasattr(myWin, 'mean_image'):
                self.feature_image = self.mean_image
            elif self.featuremap == 'Max' and hasattr(myWin, 'max_image'):
                self.feature_image = self.max_image
            self.segTh = segThread(self.feature_image, seg_ops)
            self.segTh.start()
            self.segTh.finished.connect(self.Thread_post_process)
        else:
            if hasattr(myWin, 'enhanced_image') and hasattr(myWin,'masks'):
                self.std_image = np.std(self.enhanced_image, axis=0)
                self.mean_image = np.mean(self.enhanced_image, axis=0)
                self.max_image = np.max(self.enhanced_image, axis=0)
                self.featuremap = self.comboBox_featuremap.currentText()
                if self.featuremap == 'Standard deviation' and hasattr(myWin, 'std_image'):
                    self.feature_image = self.std_image
                elif self.featuremap == 'Mean' and hasattr(myWin, 'mean_image'):
                    self.feature_image = self.mean_image
                elif self.featuremap == 'Max' and hasattr(myWin, 'max_image'):
                    self.feature_image = self.max_image
                self.labelsize_16 = [0, 0, self.frame_45.width(), self.frame_45.height()]
                self.mask_show()
                if self.all_flag:
                    self.run_extract()
            if self.checkBox_maskfix.isChecked():
                print("Mask fixed")
            return

    def seg_part(self):
        seg_ops = seg_default_ops()
        seg_ops['flow_threshold'] = float(self.lineEdit_flowthr.text())
        seg_ops['cellprob_threshold'] = float(self.lineEdit_cellprobthr.text())
        seg_ops['diameter'] = float(self.lineEdit_diameter.text())
        seg_ops['model_type'] = self.seg_model_path + '\\' + self.comboBox_segmodel.currentText()
        self.featuremap = self.comboBox_featuremap.currentText()
        if self.featuremap == 'Standard deviation' and hasattr(myWin, 'std_image'):
            self.feature_image = self.std_image
        elif self.featuremap == 'Mean' and hasattr(myWin, 'mean_image'):
            self.feature_image = self.mean_image
        elif self.featuremap == 'Max' and hasattr(myWin, 'max_image'):
            self.feature_image = self.max_image
        self.seg_partTh = seg_partThread(self.feature_image, seg_ops)
        self.seg_partTh.start()
        self.seg_partTh.finished.connect(self.Thread_post_process)

    def mask_show(self):
        # to show the roi mask after segmentation
        # To show the RGB Image in a Qlabel and adjust the contrast and brightness
        # we transform: numpy array -> PIL Image -> QImage -> QPixmap
        coloredge = np.zeros([3, self.Ly, self.Lx])  # edge pix value = 255
        self.rainbowedge = np.zeros([3, self.Ly, self.Lx])
        img_pil = Image.fromarray(self.bitconvert(self.feature_image), 'L')
        img_pil_enhance = ImageEnhance.Contrast(img_pil).enhance(self.Slider_Contrast_2.value() / 100)
        img_pil_enhance = ImageEnhance.Brightness(img_pil_enhance).enhance(self.Slider_Brightness_2.value() / 100)
        self.feature_image_enhance = np.asarray(img_pil_enhance)
        for i in range(self.n_roi):
            temp_mask = np.zeros((self.Ly, self.Lx))
            temp_mask[self.properties[i]['coords'][:,0],self.properties[i]['coords'][:,1]] = 1
            coloredge[0, :, :] = cv2.Canny(np.uint8(temp_mask), 0, 0) * self.colorma.colors[i % 11, 0]
            coloredge[1, :, :] = cv2.Canny(np.uint8(temp_mask), 0, 0) * self.colorma.colors[i % 11, 1]
            coloredge[2, :, :] = cv2.Canny(np.uint8(temp_mask), 0, 0) * self.colorma.colors[i % 11, 2]
            self.rainbowedge = self.rainbowedge + coloredge
        self.rainbowedge = np.uint8(self.rainbowedge)
        self.binaedge = np.asarray(Image.fromarray(np.uint8(self.rainbowedge.transpose(1, 2, 0)), 'RGB').convert('L'))
        self.binaedge[self.binaedge == 0] = 1
        self.binaedge[self.binaedge > 1] = 0
        rainbowmask = self.binaedge * self.feature_image_enhance + self.rainbowedge
        rainbowmask = rainbowmask.transpose(1, 2, 0)
        rainbowmask_pil = Image.fromarray(rainbowmask, 'RGB')
        # add index text
        ft = ImageFont.truetype(font='./Font/msyhbd.ttc', size=12)
        # for i in range(self.n_roi):
        #     textpen = ImageDraw.Draw(rainbowmask_pil)
        #     textpen.text((self.properties[i]['centroid'][1], self.properties[i]['centroid'][0]), str(i + 1), \
        #                  fill=tuple(np.asarray(255 * self.properties[i]['color'][0:3], dtype=int)), font=ft)
        img_Q = QtGui.QImage(rainbowmask_pil.tobytes(), rainbowmask.shape[1], rainbowmask.shape[0],
                             rainbowmask.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.pix16 = QtGui.QPixmap(img_Q)
        self.label_16.setPixmap(self.pix16)
        self.label_16.setScaledContents(True)
        self.label_16.setGeometry(
            QtCore.QRect(self.labelsize_16[0], self.labelsize_16[1], self.labelsize_16[2], self.labelsize_16[3]))
        self.label_largeseg.setPixmap(self.pix16)
        self.label_largeseg.setScaledContents(True)
        self.label_largeseg.setGeometry(
            QtCore.QRect(self.labelsize_largeseg[0], self.labelsize_largeseg[1], self.labelsize_largeseg[2], self.labelsize_largeseg[3]))

    def run_extract(self):
        # run signal extraction
        if hasattr(myWin, 'properties') and hasattr(myWin, 'enhanced_image'):
            basecor = False
            if self.checkBox_basecor.isChecked():
                basecor = True
            self.extractTh = extractThread(self.properties, self.enhanced_image, basecor)
            self.extractTh.start()
            self.extractTh.finished.connect(self.Thread_post_process)
        else:
            return

    def show_dff(self, whichtoshow):
        # to show the raw ΔF/F0 signals selected
        self.label_17.clear()
        x = np.arange(0, self.n_frames)
        for i in whichtoshow:
            i = int(i)
            self.label_17.plot(x, i + self.dff_sig[i - 1, :], pen=pyqtgraph.mkPen (255 * self.colorma.colors[(i - 1) % 11, 0:3], width= 2))
        labelStyle = {'color': 'black', 'font-size': '18pt'}
        self.label_17.setLabel('left', 'Num', **labelStyle)
        self.label_17.setLabel('bottom', 'Frames', **labelStyle)
        font = QtGui.QFont()
        font.setPixelSize(22)
        font.setBold(True)
        self.label_17.getAxis('left').setStyle(tickFont=font)
        self.label_17.getAxis('bottom').setStyle(tickFont=font)
        self.label_17.setXRange(0, self.n_frames)
        maxnumtoshow = int(np.max(whichtoshow)-1)
        minnumtoshow = int(np.min(whichtoshow))
        self.label_17.setYRange(minnumtoshow, maxnumtoshow+1+np.max(self.dff_sig[maxnumtoshow,:]))

    def choose_signal_show(self):
        # choose which signal to show
        roitext = self.lineEdit_8.text()
        roitext = roitext.split(',')
        self.roiarr = np.zeros(0)
        for i in range(len(roitext)):
            if roitext[i].isdigit() is True:
                self.roiarr = np.append(self.roiarr, int(roitext[i]))
            elif re.search(r'^[0-9]*:[0-9]*$',roitext[i]):
                temparr = roitext[i].split(':')
                if int(temparr[0])<int(temparr[1]):
                    self.roiarr = np.append(self.roiarr, np.arange(int(temparr[0]), 1+int(temparr[1]), 1))
        self.roiarr = np.sort(np.unique(self.roiarr))
        self.roiarr = self.roiarr[ ~ (self.roiarr > self.n_roi)] #Delete the num exceeding index
        self.roiarr = self.roiarr[self.roiarr > 0]
        if len(self.roiarr) != 0:
            if hasattr(myWin, 'dff_sig'):
                self.show_dff(self.roiarr)
            if hasattr(myWin, 'spk_sig'):
                self.show_spk(self.roiarr)

    def run_cas(self):
        # run spike reduction with Cascade
        if hasattr(myWin, 'dff_sig'):
            model = self.comboBox_spikemodel.currentText()
            self.casTh = casThread(self.dff_sig, model)
            self.casTh.start()
            self.casTh.finished.connect(self.Thread_post_process)
        else:
            return

    def show_spk(self, whichtoshow):
        # to show the spike selected
        self.label_18.clear()
        x = np.arange(0, self.n_frames)
        for i in whichtoshow:
            i = int(i)
            self.label_18.plot(x, i + self.spk_sig[i - 1, :], pen=pyqtgraph.mkPen (255 * self.colorma.colors[(i - 1) % 11, 0:3], width= 2))
        labelStyle = {'color': 'black', 'font-size': '18pt'}
        self.label_18.setLabel('left', 'Num', **labelStyle)
        self.label_18.setLabel('bottom', 'Frames', **labelStyle)
        font = QtGui.QFont()
        font.setPixelSize(22)
        font.setBold(True)
        self.label_18.getAxis('left').setStyle(tickFont=font)
        self.label_18.getAxis('bottom').setStyle(tickFont=font)
        self.label_18.setXRange(0, self.n_frames)
        maxnumtoshow = int(np.max(whichtoshow)-1)
        minnumtoshow = int(np.min(whichtoshow))
        self.label_18.setYRange(minnumtoshow, maxnumtoshow+1+np.max(self.dff_sig[maxnumtoshow,:]))

    def run_discrete(self):
        # if hasattr(myWin, 'spk_sig') and self.isfloat(self.lineEdit_discreterythr.text()):
        #     # discreterythr = float(self.lineEdit_discreterythr.text())
        #     self.discrete_sig = self.spk_sig.copy()
        #     self.discrete_sig = self.discrete_sig[:, 32:-32]
        #     # self.discrete_sig[self.discrete_sig > discreterythr] = 255
        #     # self.discrete_sig[self.discrete_sig <= discreterythr] = 0
        #     img_pil = Image.fromarray(np.uint8(255 * self.discrete_sig))
        #     self.pix19 = img_pil.toqpixmap()
        #     self.label_19.setPixmap(self.pix19)
        #     self.label_19.setScaledContents(True)
        # elif hasattr(myWin, 'dff_sig'):
        #     discreterythr = float(self.lineEdit_discreterythr.text())
        #     self.discrete_sig = self.dff_sig.copy()
        #     # self.discrete_sig[self.discrete_sig > discreterythr] = 255
        #     # self.discrete_sig[self.discrete_sig <= discreterythr] = 0
        #     img_pil = Image.fromarray(np.uint8(255 * self.discrete_sig))
        #     self.pix19 = img_pil.toqpixmap()
        #     self.label_19.setPixmap(self.pix19)
        #     self.label_19.setScaledContents(True)
        if  hasattr(myWin, 'spk_sig') and self.comboBox_spikemodel.currentText():
            model = self.comboBox_spikemodel.currentText()
            self.discreteTh = discreteThread(self.spk_sig, model)
            self.discreteTh.start()
            self.discreteTh.finished.connect(self.Thread_post_process)



    def run_all(self):
        if hasattr(myWin, 'raw_image') and self.isfloat(self.lineEdit_smooth.text()) and self.isfloat(self.lineEdit_maxregshift.text()) \
                and self.isfloat(self.lineEdit_smoothtime.text()) and self.isfloat(self.lineEdit_nimg_init.text()) and \
                self.isfloat(self.lineEdit_flowthr.text()) and self.isfloat(self.lineEdit_cellprobthr.text()) and \
                self.isfloat(self.lineEdit_diameter.text()) or hasattr(myWin, 'batches_count'):
            if hasattr(myWin, 'batches_count'):
                if self.batches_count != (self.filenum):
                    self._translate = QtCore.QCoreApplication.translate
                    self.clearGUI()
                    (self.file_name, ext) = os.path.splitext(os.path.basename(self.filelist[self.batches_count]))
                    self.raw_image = imread(self.folder_path + '\\' + self.filelist[self.batches_count] + '.tif')
                    self.raw_image[self.raw_image < 0] = 0
                    hist = np.histogram(self.raw_image[0, :, :], bins=256)
                    if hist[1][np.where(hist[0] == np.max(hist[0]))[0]] >= 30000:
                        self.raw_image = self.raw_image - 32768
                        self.raw_image[self.raw_image > 32768] = 0
                    print('Running pipline in file ' + str(self.batches_count + 1) + '/' + str(self.filenum))
                    hist = np.histogram(self.raw_image[0, :, :], bins=256)
                    if hist[1][np.where(hist[0] == np.max(hist[0]))[0]] >= 30000:
                        self.raw_image = self.raw_image - 32768
                        self.raw_image[self.raw_image > 32768] = 0
                    self.n_frames, self.Ly, self.Lx = self.raw_image.shape
                    self.raw_image_show = self.bitconvert(self.raw_image)
                    self.label_26.setText(self._translate("MainWindow", "1/{}".format(self.n_frames)))
                    self.horizontalScrollBar_Raw.setMaximum(self.n_frames)
                    first = self.raw_image_show[0, :, :]
                    pix = self.con_imgtopixmap(first)
                    self.label_9.setPixmap(pix)
                    self.label_9.setScaledContents(True)
                    self.labelsize_9 = [0, 0, self.frame_41.width(), self.frame_41.height()]
                    self.label_9.setGeometry(
                        QtCore.QRect(self.labelsize_9[0], self.labelsize_9[1], self.labelsize_9[2],
                                     self.labelsize_9[3]))
                    self.all_flag = 1
                    self.run_enhance()
                else:
                    self.all_flag = 0
                    self.batches_pipe = 0
                    del self.batches_count
                    print('Run batches finished')
            else:
                self.all_flag = 1
                self.run_enhance()
            # self.cal_diameter()
            # self.run_seg()
            # self.run_extract()
            # self.run_cas()
            # self.run_discrete()s
            # self.tabWidget.setCurrentWidget(self.tab_2)

    def run_batches(self):
        root = tk.Tk()  # 窗体对象
        root.withdraw()  # 窗体隐藏
        self.folder_path = filedialog.askdirectory()
        if self.folder_path == '':    # return when you did not choose a tiff file(That's when it would return a empty string to load_path)
            return
        self.batches_count = 0
        print('Reading folder:' + self.folder_path)
        self.filelist = []
        f_list = os.listdir(self.folder_path)  # 返回文件名
        for i in f_list:
            if os.path.splitext(i)[1] == '.tif':
                self.filelist.append(os.path.splitext(i)[0])
        self.filenum = len(self.filelist)
        self.batches_save_path = os.path.join(self.folder_path, "..") + '\\' + 'result_' + datetime.datetime.now().strftime('%Y%m%d')
        choose_win = Choose_win()
        self.batches_pipe = 0
        self.batches_pipe = choose_win.exec()
        if self.batches_pipe != 0:
            self.run_all()

    def isfloat(self, string):
        try:
            float(string)
            return True
        except:
            return False

    def save_masks(self):
        if hasattr(myWin, 'properties'):
            root = tk.Tk()
            root.withdraw()
            self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name)
            if self.save_path == '':
                return
            self.pix16.save(self.save_path + '_masks.png')
            # properties = list()
            # for i in range(self.n_roi):
            #     one_property = {'area':self.properties[i].area,
            #                         'centroid':self.properties[i]['centroid'],
            #                         'coords':self.properties[i]['coords']}
            #     properties.append(one_property)
            # print(properties[3])
            savemat(self.save_path + '_properties.mat',
                    {'properties': self.properties})
            print("Save masks successfully")

    def save_raw_sig(self):
        if hasattr(myWin, 'raw_sig'):
            root = tk.Tk()
            root.withdraw()
            self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name)
            if self.save_path == '':
                return
            savemat(self.save_path + '_raw_sig.mat', {'raw_sig': self.raw_sig,'dff_sig': self.dff_sig})
            dff_img = self.label_17.grab().toImage()
            dff_img.save(self.save_path + '_dff_sig.png')
            print("Save raw_sig successfully")

    def save_spk(self):
        if hasattr(myWin, 'spk_sig'):
            root = tk.Tk()
            root.withdraw()
            self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name)
            if self.save_path == '':
                return
            savemat(self.save_path + '_spk_sig.mat', {'spk_sig': self.spk_sig})
            spk_img = self.label_18.grab().toImage()
            spk_img.save(self.save_path + '_spk_sig.png')
            print("Save spk_sig successfully")

    def save_discrete(self):
        if hasattr(myWin, 'discrete_sig'):
            root = tk.Tk()
            root.withdraw()
            self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name)
            if self.save_path == '':
                return
            savemat(self.save_path + '_discrete_sig.mat', {'discrete_sig': self.discrete_sig,'spike_time_estimates': self.spike_time_estimates})
            self.pix19.save(self.save_path + '_discrete_sig.png')
            print("Save discrete_sig successfully")

    def save_all(self):
        root = tk.Tk()
        root.withdraw()
        self.save_path = filedialog.asksaveasfilename(initialfile=self.file_name)
        if self.save_path == '':
            return
        result = dict()
        if hasattr(myWin, 'enhanced_image'):
           with TiffWriter(self.save_path + '_enhanced.tif', bigtiff=True) as tif:
               tif.write(np.fix(self.enhanced_image).astype(np.int16))
        if hasattr(myWin, 'rainbowedge'):
            self.pix16.save(self.save_path + '_masks.png')
        if hasattr(myWin, 'feature_image'):
            result.update({'feature_image': self.feature_image})
        if hasattr(myWin, 'properties'):
            result.update( {'properties': self.properties})
        if hasattr(myWin, 'raw_sig'):
            result.update({'raw_sig': self.raw_sig})
            result.update({'dff_sig': self.dff_sig})
            dff_img = self.label_17.grab().toImage()
            dff_img.save(self.save_path + '_dff_sig.png')
        if hasattr(myWin, 'spk_sig'):
            result.update({'spk_sig': self.spk_sig})
            spk_img = self.label_18.grab().toImage()
            spk_img.save(self.save_path + '_spk_sig.png')
        if hasattr(myWin, 'discrete_sig'):
            result.update({'discrete_sig': self.discrete_sig})
            self.pix19.save(self.save_path + '_discrete_sig.png')
        if hasattr(myWin, 'spike_time_estimates'):
            result.update({'spike_time_estimates': self.spike_time_estimates})
        savemat(self.save_path + '.mat',result)
        print("Save all finished")
        return

    def Thread_post_process(self):
        #This function is used for post-processing after data processing in different threads is complete
        if hasattr(myWin, 'enhanceTh'):
            self.enhanced_image, self.std_image, self.mean_image , self.max_image= self.enhanceTh.get_result()
            del self.enhanceTh
            self.enhanced_image_show = self.bitconvert(self.enhanced_image)
            self.horizontalScrollBar_Enhance.setMaximum(self.n_frames)
            if self.all_flag:
                if self.batches_pipe == 1:
                    self.save_batches()
                    self.batches_count += 1
                    self.run_all()
                else:
                    self.run_seg()
        if hasattr(myWin, 'segTh'):
            try:
                mask = self.segTh.get_result()
            except:
                del self.segTh
                return
            del self.segTh
            properties = measure.regionprops(mask)
            self.n_roi = len(properties)
            if self.n_roi != 0:
                self.properties = list()
            for i in range(self.n_roi):
                one_property = {'area': properties[i].area,
                                'centroid': np.array(properties[i].centroid),
                                'coords': properties[i].coords,
                                'color': self.colorma.colors[i % 11,:]}
                self.properties.append(one_property)
            if self.n_roi != 0:
                self.labelsize_16 = [0, 0, self.frame_45.width(), self.frame_45.height()]
                self.mask_show()
                self.label_n_roi.setText(str(self.n_roi))
                self.label_n_roi_2.setText(str(self.n_roi))
                print('Segmentation finished')
            else:
                print('There is no ROI !')
            if self.all_flag:
                if self.batches_pipe == 2:
                    self.save_batches()
                    self.batches_count += 1
                    self.run_all()
                else:
                    self.run_extract()
        if hasattr(myWin, 'seg_partTh'):
            try:
                mask = self.seg_partTh.get_result()
            except:
                del self.seg_partTh
                return
            del self.seg_partTh
            properties = measure.regionprops(mask)
            self.n_roi = len(properties)
            if self.n_roi != 0:
                self.properties = list()
            for i in range(self.n_roi):
                one_property = {'area': properties[i].area,
                                'centroid': np.array(properties[i].centroid),
                                'coords': properties[i].coords,
                                'color': self.colorma.colors[i % 11,:]}
                self.properties.append(one_property)
            if self.n_roi != 0:
                self.labelsize_16 = [0, 0, self.frame_45.width(), self.frame_45.height()]
                self.mask_show()
                self.label_n_roi.setText(str(self.n_roi))
                self.label_n_roi_2.setText(str(self.n_roi))
                print('Segmentation finished')
            else:
                print('There is no ROI !')
            if self.all_flag:
                if self.batches_pipe == 2:
                    self.save_batches()
                    self.batches_count += 1
                    self.run_all()
                else:
                    self.run_extract()
        if hasattr(myWin, 'extractTh'):
            try:
                self.raw_sig, self.dff_sig = self.extractTh.get_result()
            except:
                del self.extractTh
                return
            del self.extractTh
            # Show some of the first extracted signals in plotWidget
            if hasattr(myWin, 'roiarr') and len(self.roiarr) != 0:
                self.show_dff(self.roiarr)
            elif self.n_roi > 10:
                self.show_dff(range(1, 11))
            else:
                self.show_dff(range(self.n_roi))
            if self.all_flag:
                if self.batches_pipe == 3:
                    self.save_batches()
                    self.batches_count += 1
                    self.run_all()
                else:
                    self.run_cas()
        if hasattr(myWin, 'casTh'):
            try:
                self.spk_sig = self.casTh.get_result()
            except:
                del self.casTh
                return
            del self.casTh
            if hasattr(myWin, 'roiarr') and len(self.roiarr) != 0:
                self.show_spk(self.roiarr)
            elif self.n_roi > 10:
                self.show_spk(range(1, 11))
            else:
                self.show_spk(range(self.n_roi))
            if self.all_flag:
                if self.batches_pipe == 4:
                    self.save_batches()
                    self.batches_count += 1
                    self.run_all()
                else:
                    self.run_discrete()
        if hasattr(myWin, 'discreteTh'):
            try:
                self.discrete_sig, self.spike_time_estimates = self.discreteTh.get_result()
            except:
                del self.discreteTh
                return
            del self.discreteTh
            self.discrete_sig = self.discrete_sig[:, 32:-32] * 255
            self.discrete_sig_sorted = self.Cor_sort(self.discrete_sig)
            img_pil = Image.fromarray(np.uint8(self.discrete_sig_sorted))
            self.pix19 = img_pil.toqpixmap()
            self.label_19.setPixmap(self.pix19)
            self.label_19.setScaledContents(True)
            if self.all_flag:
                self.all_flag = 0
                self.save_batches()
                self.batches_count = self.batches_count + 1
                self.run_all()

    def save_batches(self):
        if hasattr(myWin, 'batches_count'):
            if not os.path.exists(self.batches_save_path):
                os.makedirs(self.batches_save_path)
            single_save_path = self.batches_save_path + '\\' + self.filelist[self.batches_count]
            os.makedirs(single_save_path)
            with TiffWriter(single_save_path + '\\' + self.filelist[self.batches_count] + '_Enhanced.tif',
                            bigtiff=True) as tif:
                tif.write(np.fix(self.enhanced_image).astype(np.int16))
            if self.batches_pipe == 0:
                return
            elif self.batches_pipe == 2:
                result = {'properties': self.properties,'feature_image': self.feature_image}
                self.pix16.save(single_save_path + '\\' + self.filelist[self.batches_count] + '_masks.png')
                savemat(single_save_path + '\\' + self.filelist[self.batches_count] + '_result.mat', result)
            elif self.batches_pipe == 3:
                result = {'properties': self.properties, 'raw_sig': self.raw_sig, 'dff_sig': self.dff_sig,
                         'feature_image': self.feature_image}
                self.pix16.save(single_save_path + '\\' + self.filelist[self.batches_count] + '_masks.png')
                savemat(single_save_path + '\\' + self.filelist[self.batches_count] + '_result.mat', result)
            elif self.batches_pipe == 4:
                result = {'properties': self.properties, 'raw_sig': self.raw_sig, 'dff_sig': self.dff_sig,
                          'spk_sig': self.spk_sig,'feature_image': self.feature_image}
                self.pix16.save(single_save_path + '\\' + self.filelist[self.batches_count] + '_masks.png')
                savemat(single_save_path + '\\' + self.filelist[self.batches_count] + '_result.mat', result)
            elif self.batches_pipe == 5:
                result = {'properties': self.properties, 'raw_sig': self.raw_sig, 'dff_sig': self.dff_sig,
                          'spk_sig': self.spk_sig, 'discrete_sig': self.discrete_sig,
                          'feature_image': self.feature_image,
                          'spike_time_estimates': self.spike_time_estimates}
                self.pix16.save(single_save_path + '\\' + self.filelist[self.batches_count] + '_masks.png')
                savemat(single_save_path + '\\' + self.filelist[self.batches_count] + '_result.mat', result)

    def clearGUI(self):
        if hasattr(myWin, 'raw_image'):
            del self.raw_image
        if hasattr(myWin, 'enhanced_image'):
            del self.enhanced_image
        if hasattr(myWin, 'feature_image'):
            del self.feature_image
        if hasattr(myWin, 'properties') and not self.checkBox_maskfix.isChecked():
            self.label_n_roi.setText('∞')
            self.label_n_roi_2.setText('∞')
            del self.properties
        if hasattr(myWin,'rainbowedge'):
            del self.rainbowedge
        if hasattr(myWin, 'raw_sig'):
            del self.raw_sig
        if hasattr(myWin, 'dff_sig'):
            del self.dff_sig
        if hasattr(myWin, 'spk_sig'):
            del self.spk_sig
        if hasattr(myWin, 'discrete_sig'):
            del self.discrete_sig
        self.label_9.clear()
        self.label_10.clear()
        self.label_16.clear()
        self.label_17.clear()
        self.label_18.clear()
        self.label_19.clear()

    def Cor_sort(self, Signal):
        pointnum = 100;
        times = 30;
        roinum, sig_len = Signal.shape
        i = 0
        tempnum = roinum
        while i < tempnum:
            if np.max(Signal[i,:]) == 0:
                Signal = np.delete(Signal,i,axis=0)
                roinum -= 1
                i -=1
                tempnum -= 1
            i = i+1
        sumsig = np.sum(Signal, 0)
        timespk = np.array(list(map(list(sumsig).index,heapq.nlargest(pointnum,sumsig))))
        template = np.zeros(sig_len)
        template[timespk] = 255
        cormartix = np.zeros(roinum)
        for i in range(roinum):
            cormartix[i] = np.min(np.corrcoef(Signal[i,:], template))
        index = np.argsort(cormartix)
        sig_sorted = np.zeros([roinum, sig_len])
        for i in range(roinum):
            sig_sorted[i,:]=Signal[index[i],:]
        for i in range(times):
            sig_temp = sig_sorted
            for j in range(roinum):
                cormartix[j] = np.min(np.corrcoef(sig_sorted[j,:], np.mean(sig_temp[0:9,:], 0)))
            index = np.argsort(cormartix)
            sig_sorted = np.zeros([roinum, sig_len])
            for j in range(roinum):
                sig_sorted[j,:] = sig_temp[index[j],:]
        return sig_sorted

class enhanceThread(QThread):

    def __init__(self, raw_image, reg_ops, algorithm, model, parent=None):
        super(enhanceThread, self).__init__(parent)
        # 获取初始输入值
        self.raw_image = raw_image
        self.reg_ops = reg_ops
        self.algorithm = algorithm
        self.model = model
        self.enhanced_image = None
        self.feature_image = None

    def run(self):
        #reg_image = self.raw_image
        reg_image = reg(self.raw_image.astype(np.int16), self.reg_ops)
        #self.enhanced_image = reg_image
        if self.algorithm == 'DeepCAD_RT':
            self.enhanced_image = deep_denoise(reg_image, self.model)
        elif self.algorithm == 'SRDTrans':
            self.enhanced_image = srd_denoise(reg_image, self.model)
        self.std_image = np.std(self.enhanced_image, axis=0)
        self.mean_image = np.mean(self.enhanced_image, axis=0)
        self.max_image = np.max(self.enhanced_image, axis=0)

    def get_result(self):
        return self.enhanced_image, self.std_image, self.mean_image, self.max_image

class segThread(QThread):

    def __init__(self, feature_image, seg_ops, parent=None):
        super(segThread, self).__init__(parent)
        self.feature_image = feature_image
        self.seg_ops = seg_ops

    def run(self):
        self.mask = seg(self.feature_image , self.seg_ops)

    def get_result(self):
        return self.mask

class seg_partThread(QThread):

    def __init__(self, feature_image, seg_ops, parent=None):
        super(seg_partThread, self).__init__(parent)
        self.feature_image = feature_image
        self.seg_ops = seg_ops

    def run(self):
        sz = self.feature_image.shape[1]
        bias = 0
        if sz == 2048:
            self.mask_all = np.zeros([self.feature_image.shape[0],self.feature_image.shape[1]])
            for i in range(4):
                for j in range(4):
                    print('seg_part in '+str(i*4+j+1))
                    feature_image_part = self.feature_image[i*512:(i+1)*512,j*512:(j+1)*512]
                    mask_part = seg(feature_image_part , self.seg_ops)
                    self.mask_all[i*512:(i+1)*512,j*512:(j+1)*512] = mask_part+bias*(mask_part>0)
                    bias = np.max(self.mask_all)
            print(np.max(self.mask_all))
        elif sz == 4096:
            self.mask_all = np.zeros([self.feature_image.shape[0], self.feature_image.shape[1]])
            for i in range(4):
                for j in range(4):
                    print('seg_part in ' + str(i * 4 + j + 1))
                    feature_image_part = self.feature_image[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024]
                    mask_part = seg(feature_image_part, self.seg_ops)
                    self.mask_all[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024] =  mask_part+bias*(mask_part>0)
                    bias = np.max(self.mask_all)
            print( np.max(self.mask_all))
        else:
            print('sz is not 2048 or 4096')

    def get_result(self):
        return self.mask_all.astype(np.int16)

class extractThread(QThread):

    def __init__(self, properties, enhanced_image, basecor, parent=None):
        super(extractThread, self).__init__(parent)
        self.properties = properties
        self.enhanced_image = enhanced_image
        self.basecor = basecor

    def run(self):
        n_roi = len(self.properties)
        n_frames = self.enhanced_image.shape[0]
        self.raw_sig = np.zeros([n_roi, n_frames])
        self.dff_sig = np.zeros([n_roi, n_frames])
        if self.basecor is True:
            t = np.arange(0,n_frames)
            for i in trange(n_roi, colour='blue', desc='Signal extracting(Baseline correction)'):
                # self.raw_sig[i, :] = np.sum(self.masks[i] * self.enhanced_image, axis=(1, 2))
                self.raw_sig[i, :] = np.sum(self.enhanced_image[:, self.properties[i]['coords'][:, 0],
                                            self.properties[i]['coords'][:, 1]], axis=1)
                baseline = csaps(t, self.raw_sig[i, :], t, smooth=1e-8)
                self.dff_sig[i, :] = (self.raw_sig[i, :] - baseline) / baseline
        else:
            for i in trange(n_roi, colour='blue', desc='Signal extracting'):
                #self.raw_sig[i, :] = np.sum(self.masks[i] * self.enhanced_image, axis=(1, 2))
                self.raw_sig[i, :] = np.sum(self.enhanced_image[:, self.properties[i]['coords'][:, 0],
                                            self.properties[i]['coords'][:, 1]], axis=1)
                sort_sig = sorted(self.raw_sig[i, :])
                baseline = np.mean(sort_sig[int(0.10 * n_frames): int(0.30 * n_frames)])
                self.dff_sig[i, :] = (self.raw_sig[i, :] - baseline) / baseline

    def get_result(self):
        return self.raw_sig, self.dff_sig

class casThread(QThread):

    def __init__(self, dff_sig, model, parent=None):
        super(casThread, self).__init__(parent)
        self.dff_sig = dff_sig
        self.model = model

    def run(self):
        self.spk_sig = cas(self.dff_sig, self.model)

    def get_result(self):
        return self.spk_sig


class discreteThread(QThread):

    def __init__(self, spk_sig, model, parent=None):
        super(discreteThread, self).__init__(parent)
        self.spk_sig = spk_sig
        self.model = model

    def run(self):
        self.discrete_sig,  self.spike_time_estimates = infer_discrete_spikes(self.spk_sig ,self.model)

    def get_result(self):
        return self.discrete_sig,  self.spike_time_estimates


class Choose_win(Qt.QDialog):
    def __init__(self, parent=None):
        super(Choose_win, self).__init__(parent)
        self.setWindowTitle("Select Pipeline")
        self.setStyleSheet(
                            "font:13px \"Microsoft YaHei\";\n"
                            "border-image:none;\n"
                            "\n"
                            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(128, 218, 205, 255), stop:1 rgba(44, 175, 255, 158));\n"
                            )
        groupBox = Qt.QGroupBox("Pipe")
        groupBox.setFlat(False)
        self.checkBox_enhance = Qt.QCheckBox("Enhance")
        self.checkBox_seg = Qt.QCheckBox("Seg")
        self.checkBox_extract = Qt.QCheckBox("Extract")
        self.checkBox_cas = Qt.QCheckBox("Cas")
        self.checkBox_discrete = Qt.QCheckBox("Discrete")
        self.pushButton_confirm = Qt.QPushButton("confirm")
        self.pushButton_cancel = Qt.QPushButton("cancel")
        self.checkBox_enhance.clicked.connect(lambda : self.checkstate("enhance"))
        self.checkBox_seg.clicked.connect(lambda : self.checkstate("seg"))
        self.checkBox_extract.clicked.connect(lambda : self.checkstate("extract"))
        self.checkBox_cas.clicked.connect(lambda : self.checkstate("cas"))
        self.checkBox_discrete.clicked.connect(lambda : self.checkstate("discrete"))
        layout = Qt.QHBoxLayout()
        layout.addWidget(self.checkBox_enhance)
        layout.addWidget(self.checkBox_seg)
        layout.addWidget(self.checkBox_extract)
        layout.addWidget(self.checkBox_cas)
        layout.addWidget(self.checkBox_discrete)
        groupBox.setLayout(layout)
        mainLayout = Qt.QVBoxLayout()
        mainLayout.addWidget(groupBox)
        groupBox.setStyleSheet("QCheckBox{\n background:transparent;\n}")
        vbox = Qt.QVBoxLayout()
        hbox = Qt.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.pushButton_confirm)
        hbox.addWidget(self.pushButton_cancel)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        mainLayout.addLayout(vbox)
        self.setLayout(mainLayout)
        self.pushButton_confirm.clicked.connect(lambda : self.closeDialog("confirm"))
        self.pushButton_cancel.clicked.connect(lambda : self.closeDialog("cancel"))

    def checkstate(self, which_choose):
        if which_choose == "enhance":
            self.checkBox_enhance.setChecked(True)
            self.checkBox_seg.setChecked(False)
            self.checkBox_extract.setChecked(False)
            self.checkBox_cas.setChecked(False)
            self.checkBox_discrete.setChecked(False)
        if which_choose == "seg":
            self.checkBox_enhance.setChecked(True)
            self.checkBox_seg.setChecked(True)
            self.checkBox_extract.setChecked(False)
            self.checkBox_cas.setChecked(False)
            self.checkBox_discrete.setChecked(False)
        if which_choose == "extract":
            self.checkBox_enhance.setChecked(True)
            self.checkBox_seg.setChecked(True)
            self.checkBox_extract.setChecked(True)
            self.checkBox_cas.setChecked(False)
            self.checkBox_discrete.setChecked(False)
        if which_choose == "cas":
            self.checkBox_enhance.setChecked(True)
            self.checkBox_seg.setChecked(True)
            self.checkBox_extract.setChecked(True)
            self.checkBox_cas.setChecked(True)
            self.checkBox_discrete.setChecked(False)
        if which_choose == "discrete":
            self.checkBox_enhance.setChecked(True)
            self.checkBox_seg.setChecked(True)
            self.checkBox_extract.setChecked(True)
            self.checkBox_cas.setChecked(True)
            self.checkBox_discrete.setChecked(True)

    def closeDialog(self, order):
        if order == "confirm":
            if self.checkBox_discrete.isChecked() == True:
                self.done(5)
            elif self.checkBox_cas.isChecked() == True:
                self.done(4)
            elif self.checkBox_extract.isChecked() == True:
                self.done(3)
            elif self.checkBox_seg.isChecked() == True:
                self.done(2)
            elif self.checkBox_enhance.isChecked() == True:
                self.done(1)
            else:
                self.done(0)
        if order == "cancel":
            self.done(0)

if __name__ == "__main__":
    # load QApplication
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    # initialization
    myWin = NIPWindow()
    # show window
    myWin.show()
    # exit
    sys.exit(app.exec_())
