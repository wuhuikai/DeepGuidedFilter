import torch
import torch.nn as nn
import torch.nn.parallel

from guided_filter_pytorch.guided_filter import GuidedFilter

class network_dss(nn.Module):
    def __init__(self, input_nc, dgf, dgf_r, dgf_eps, post_sigmoid):
        super(network_dss, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(input_nc, 64, 3, 1, 35))
        conv1.add_module('relu1_1', nn.ReLU())
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU())
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3', nn.AvgPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.AvgPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        dsn6 = nn.Sequential()
        dsn6.add_module('pool5', nn.AvgPool2d(2, 2))
        dsn6.add_module('dsn6_conv1', nn.Conv2d(512, 512, 7, 1, 3))
        dsn6.add_module('dsn6_relu1', nn.ReLU())
        dsn6.add_module('dsn6_conv2', nn.Conv2d(512, 512, 7, 1, 3))
        dsn6.add_module('dsn6_relu2', nn.ReLU())
        dsn6.add_module('dsn6_conv3', nn.Conv2d(512, 1, 1))
        dsn6.add_module('upsample_32', nn.ConvTranspose2d(1, 1, 64, 32))
        self.dsn6 = dsn6

        dsn5 = nn.Sequential()
        dsn5.add_module('dsn5_conv1', nn.Conv2d(512, 512, 5, 1, 2))
        dsn5.add_module('dsn5-relu1', nn.ReLU())
        dsn5.add_module('dsn5-conv2', nn.Conv2d(512, 512, 5, 1, 2))
        dsn5.add_module('dsn5-relu2', nn.ReLU())
        dsn5.add_module('dsn5-conv3', nn.Conv2d(512, 1, 1))
        dsn5.add_module('upsample_16', nn.ConvTranspose2d(1, 1, 32, 16))
        self.dsn5 = dsn5

        dsn4 = nn.Sequential()
        dsn4.add_module('dsn4_conv1', nn.Conv2d(512, 256, 5, 1, 2))
        dsn4.add_module('dsn4-relu1', nn.ReLU())
        dsn4.add_module('dsn4-conv2', nn.Conv2d(256, 256, 5, 1, 2))
        dsn4.add_module('dsn4-relu2', nn.ReLU())
        dsn4.add_module('dsn4-conv3', nn.Conv2d(256, 1, 1))
        dsn4.add_module('upsample_8', nn.ConvTranspose2d(1, 1, 16, 8))
        self.dsn4 = dsn4

        dsn3 = nn.Sequential()
        dsn3.add_module('dsn3_conv1', nn.Conv2d(256, 256, 5, 1, 2))
        dsn3.add_module('dsn3-relu1', nn.ReLU())
        dsn3.add_module('dsn3-conv2', nn.Conv2d(256, 256, 5, 1, 2))
        dsn3.add_module('dsn3-relu2', nn.ReLU())
        dsn3.add_module('dsn3-conv3', nn.Conv2d(256, 1, 1))
        dsn3.add_module('upsample_4', nn.ConvTranspose2d(1, 1, 8, 4))
        self.dsn3 = dsn3

        dsn2 = nn.Sequential()
        dsn2.add_module('dsn2_conv1', nn.Conv2d(128, 128, 3, 1, 1))
        dsn2.add_module('dsn2-relu1', nn.ReLU())
        dsn2.add_module('dsn2-conv2', nn.Conv2d(128, 128, 3, 1, 1))
        dsn2.add_module('dsn2-relu2', nn.ReLU())
        dsn2.add_module('dsn2-conv3', nn.Conv2d(128, 1, 1))
        dsn2.add_module('upsample_2', nn.ConvTranspose2d(1, 1, 4, 2))
        self.dsn2 = dsn2

        dsn1 = nn.Sequential()
        dsn1.add_module('dsn1_conv1', nn.Conv2d(64, 128, 3, 1, 1))
        dsn1.add_module('dsn1-relu1', nn.ReLU())
        dsn1.add_module('dsn1-conv2', nn.Conv2d(128, 128, 3, 1, 1))
        dsn1.add_module('dsn1-relu2', nn.ReLU())
        dsn1.add_module('dsn1-conv3', nn.Conv2d(128, 1, 1))
        self.dsn1 = dsn1

        self.upscore_dsn4 = nn.Conv2d(3, 1, 1)
        self.upscore_dsn3 = nn.Conv2d(3, 1, 1)
        self.upscore_dsn2 = nn.Conv2d(5, 1, 1)
        self.upscore_dsn1 = nn.Conv2d(5, 1, 1)
        self.upscore_fuse = nn.Conv2d(5, 1, 1)

        self.sig = nn.Sigmoid()

        # DGF
        self.dgf = dgf

        if self.dgf:
            self.guided_map_conv1 = nn.Conv2d(3,  64, 1)
            self.guided_map_relu1 = nn.ReLU(inplace=True)
            self.guided_map_conv2 = nn.Conv2d(64,  1, 1)

            self.guided_filter = GuidedFilter(dgf_r, dgf_eps)

        self.post_sigmoid = post_sigmoid


    def crop_layer(self, x, target):
        crop_h = (x.size(2) - target.size(2) + 1) // 2
        crop_w = (x.size(3) - target.size(3) + 1) // 2
        y = x[:, :, crop_h:crop_h + target.size(2), crop_w:crop_w + target.size(3)]

        return y

    def forward(self, input, x):
        out_conv1 = self.conv1(input)
        dsn1_smap = self.dsn1(out_conv1)
        dsn1_crop = self.crop_layer(dsn1_smap, input)

        out_conv2 = self.conv2(out_conv1)
        dsn2_smap = self.dsn2(out_conv2)
        dsn2_crop = self.crop_layer(dsn2_smap, input)

        out_conv3 = self.conv3(out_conv2)
        dsn3_smap = self.dsn3(out_conv3)
        dsn3_crop = self.crop_layer(dsn3_smap, input)

        out_conv4 = self.conv4(out_conv3)
        dsn4_smap = self.dsn4(out_conv4)
        dsn4_crop = self.crop_layer(dsn4_smap, input)

        out_conv5 = self.conv5(out_conv4)
        dsn5_smap = self.dsn5(out_conv5)
        dsn5_crop = self.crop_layer(dsn5_smap, input)

        dsn6_smap = self.dsn6(out_conv5)
        dsn6_crop = self.crop_layer(dsn6_smap, input)

        # fusion
        dsn4_fuse_input = torch.cat((dsn5_crop, dsn6_crop, dsn4_crop), 1)
        output_dsn4 = self.upscore_dsn4(dsn4_fuse_input)

        dsn3_fuse_input = torch.cat((dsn5_crop, dsn6_crop, dsn3_crop), 1)
        output_dsn3 = self.upscore_dsn3(dsn3_fuse_input)

        dsn2_fuse_input = torch.cat((output_dsn3, output_dsn4, dsn5_crop, dsn6_crop, dsn2_crop), 1)
        output_dsn2 = self.upscore_dsn2(dsn2_fuse_input)

        dsn1_fuse_input = torch.cat((output_dsn3, output_dsn4, dsn5_crop, dsn6_crop, dsn1_crop), 1)
        output_dsn1 = self.upscore_dsn1(dsn1_fuse_input)

        fuse_input = torch.cat((output_dsn1, output_dsn2, output_dsn3, output_dsn4, dsn5_crop), 1)

        output = self.upscore_fuse(fuse_input)
        side_out1 = self.sig(output_dsn1)
        side_out2 = self.sig(output_dsn2)
        side_out3 = self.sig(output_dsn3)
        side_out4 = self.sig(output_dsn4)
        side_out5 = self.sig(dsn5_crop)
        side_out6 = self.sig(dsn6_crop)

        if not self.post_sigmoid:
            output = self.sig(output)

        if self.dgf:
            g = self.guided_map_relu1(self.guided_map_conv1(x))
            g = self.guided_map_conv2(g)

            output = self.guided_filter(g, output)

            if not self.post_sigmoid:
                output = output.clamp(0, 1)

        if self.post_sigmoid:
            output = self.sig(output)

        return output, side_out1, side_out2, side_out3, side_out4, side_out5, side_out6
