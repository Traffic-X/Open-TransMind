import paddle
from paddle import nn


class CBAM_Module(nn.Layer):  
    def __init__(self, channels, reduction=16):  
        super(CBAM_Module, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)  
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)  
        self.fc1 = nn.Conv2D(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Conv2D(in_channels=channels // reduction, out_channels=channels, kernel_size=1, padding=0)  

        self.sigmoid_channel = nn.Sigmoid()  
        self.conv_after_concat = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  
        self.sigmoid_spatial = nn.Sigmoid()  

    def forward(self, x):  
        # Channel Attention Module  
        module_input = x  
        avg = self.relu(self.fc1(self.avg_pool(x)))  
        avg = self.fc2(avg)  
        mx = self.relu(self.fc1(self.max_pool(x)))  
        mx = self.fc2(mx)  
        x = avg + mx  
        x = self.sigmoid_channel(x)  

        # Spatial Attention Module  
        x = module_input * x  
        module_input = x  
        avg = paddle.mean(x, axis=1, keepdim=True)  
        mx = paddle.argmax(x, axis=1, keepdim=True)
        mx = paddle.cast(mx, 'float32')
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)  
        x = self.sigmoid_spatial(x)  
        x = module_input * x  

        return x 