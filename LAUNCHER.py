from UI import UI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys,os,shutil,torch
import UTILS as utils
import PIL.Image as Image
import numpy as np
from NETWORK import *

class Launcher(QWidget,UI):
    def __init__(self):
        utils.logMaker('INFO','APPLICATION LAUNCHED')

        super(Launcher,self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(),self.height())
        self.setWindowIcon(QIcon(config.icon_path))
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.open.clicked.connect(self.callManage)
        self.denoising.clicked.connect(self.callManage)
        self.save.clicked.connect(self.callManage)
        self.exit.clicked.connect(self.callManage)
        self.timer.timeout.connect(self.tipClose)

        self.thumbnailPath=None
        self.filePath=None
        self.denoised_thumbnailPath=None
        self.denoisedPath=None
        self.denoised_fileName=None

    def callManage(self):
        if self.sender()==self.open:
            filePath = QFileDialog.getOpenFileName(self,'选择文件','','*.jpg;*.jpeg;*.png')[0].replace('/','\\')
            if filePath=='':
                return

            utils.logMaker('INFO','FILE CHOSEN',[filePath])
            self.thumbnailPath=None
            self.filePath=None
            self.denoised_thumbnailPath=None
            self.denoisedPath=None
            self.denoised_fileName=None
            utils.logMaker('INFO','CACHE DELETED',os.listdir(config.cache_dir))
            for cache in os.listdir(config.cache_dir):
                os.remove(os.path.join(config.cache_dir,cache))

            self.filePath=filePath
            self.thumbnailPath=utils.thumbnail(filePath)
            utils.logMaker('INFO','THUMBNAIL FILE CREATED',[self.thumbnailPath])
            self.orgPic.clear()
            self.denoisedPic.clear()
            self.pixOrg=QPixmap(self.thumbnailPath)
            self.orgPic.setPixmap(self.pixOrg)

        if self.sender()==self.denoising:
            if self.filePath==None:
                utils.logMaker('WARRING','FILE NOT CHOSEN')
                self.saveTip.setText('未选择文件')
                self.saveTip.setVisible(True)
                self.timer.start(1500)
            else:
                self.buttonDisable.show()
                self.saveTip.setVisible(True)
                self.denoisedPath=self.runNet(self.filePath)
                self.saveTip.setText('处理完成')
                self.timer.start(1500)

                self.denoised_fileName=self.denoisedPath.split('\\')[-1]
                self.denoised_thumbnailPath=utils.thumbnail(self.denoisedPath)
                utils.logMaker('INFO','DENOISED THUMBNAIL FILE CREATED',[self.denoised_thumbnailPath])
                self.denoisedPic.clear()
                self.pixDen=QPixmap(self.denoised_thumbnailPath)
                self.denoisedPic.setPixmap(self.pixDen)

        if self.sender()==self.save:
            if self.denoisedPath==None:
                utils.logMaker('WARRING','NETWORK NOT LAUNCH YET')
                self.saveTip.setText('未处理')
                self.saveTip.setVisible(True)
                self.timer.start(1500)
            else:
                savePath=QFileDialog.getSaveFileName(self,'保存','{}'.format(self.denoised_fileName),'*.jpg;;*.jpeg;;*.png')[0].replace('/','\\')
                if savePath=='':
                    return

                saveDir=savePath.rsplit('\\',1)[0]
                shutil.copy(self.denoisedPath,saveDir)
                os.rename(os.path.join(saveDir,self.denoised_fileName),savePath)
                utils.logMaker('INFO','DENOISED FILE SAVED',[savePath])
                self.saveTip.setText('保存成功')
                self.saveTip.setVisible(True)
                self.timer.start(1500)

        if self.sender()==self.exit:
            self.close()
            utils.logMaker('INFO','APPLICATION CLOSED')

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.mouseFlag=True
            self.mousePosition=event.globalPos()-self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.mouseFlag:
            self.move(QMouseEvent.globalPos()-self.mousePosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.mouseFlag=False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def tipClose(self):
        self.saveTip.setVisible(False)
        self.buttonDisable.close()
        self.timer.stop()

    def netInit(self):
        utils.logMaker('INFO','NETWORK FILES INITIALIZING...')
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_srgan=torch.load(config.network_srg_path,map_location=self.device.type)
        self.generator_idgan=torch.load(config.network_idg_path,map_location=self.device.type)

    def runNet(self,path):
        fileName=path.split('\\')[-1]
        data=torch.tensor(np.array(Image.open(path).convert('RGB'),dtype=np.float32).transpose([2,0,1])/255-0.5).unsqueeze(dim=0).to(self.device)
        self.saveTip.setText('网络计算中....')
        data_denoised=self.generator_idgan(data)
        data_upscale=self.generator_srgan(data_denoised)

        utils.logMaker('INFO','OPERATION SUCCESSFULLY')

        pic_array=(data_upscale[0].cpu().detach().numpy()+0.5)*255
        picDenoised=Image.fromarray(pic_array.transpose([1,2,0]).astype(np.uint8))
        cachePath=os.path.join(config.cache_dir,'denoised_{}'.format(fileName))
        picDenoised.save(cachePath)

        utils.logMaker('INFO','DENOISED FILE SAVED IN CACHE',[cachePath])

        return cachePath

    def completion(self):
        check=utils.completionCheck()
        if not check[0]:
            self.ok=QPushButton("确定")
            self.ok.setStyleSheet("background-color:rgb(110,200,209);color:white;")

            self.tipBox=QMessageBox()
            self.tipBox.setWindowFlags(Qt.FramelessWindowHint)
            self.tipBox.setText("文件缺失")
            self.tipBox.setWindowTitle("提示")
            self.tipBox.setStyleSheet("background-color:rgb(51,51,51);color:white;")
            self.tipBox.addButton(self.ok,QMessageBox.AcceptRole)
            self.tipBox.setIcon(QMessageBox.NoIcon)
            self.tipBox.show()
            utils.logMaker('ERROR','FILES NOT EXIST',check[1])
            utils.logMaker('INFO','EXCEPTION CLOSED')
        else:
            self.netInit()
            self.show()

if __name__=="__main__":
    app=QApplication(sys.argv)
    launcher=Launcher()
    launcher.completion()
    app.exec_()
    utils.logMaker('INFO','CACHE DELETED',os.listdir(config.cache_dir))
    for cache in os.listdir(config.cache_dir):
        os.remove(os.path.join(config.cache_dir,cache))
