import pydicom
import numpy as np
# from PIL import Image

def readdcm(path):
    """
    :param path: 输入文件
    :return: 输出图片
    """
    ds = pydicom.dcmread(path, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    data = ds.pixel_array
    b = int(ds.RescaleIntercept)
    k = float(ds.RescaleSlope)
    data = k * data + b


    # data1 = np.array(data1, dtype='uint16')
    # imag = imag.convert('L')
    # imag = Image.merge('RGB', (imag, imag, imag))


    data = np.where(data < -1024, -1024, data)
    return data



def writedcm(mypath,path,data):
    """
    :param mypath: 副本路径
    :param path: 输出路径
    :param data: 灰度信息
    :return:
    """
    ds = pydicom.dcmread(mypath, force=True)
    #ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    b = int(ds.RescaleIntercept)
    k = float(ds.RescaleSlope)
    data = np.array((data - b) / k, dtype=np.int16)
    ds.PixelData = data
    ds.Columns = data.shape[1]
    ds.Rows = data.shape[0]
    ds.PixelSpacing = [0.04, 0.04]
    ds.save_as(path)