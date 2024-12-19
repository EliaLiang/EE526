
import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80, backbone = "ghostnet").to(device)
    summary(m, input_size=(3, 416, 416))
    
    # mobilenetv1-yolov4 40,952,893
    # mobilenetv2-yolov4 39,062,013
    # mobilenetv3-yolov4 39,989,933

    # change panet of mobilenetv1-yolov4 12,692,029
    # change panet of mobilenetv2-yolov4 10,801,149
    # change panet of mobilenetv3-yolov4 11,729,069

