import torch
import torch.nn as nn


def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                   norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU())


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3,
                                    norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            #norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class EnvironmentModel(nn.Module):
    def __init__(self, opt=None):
        super(EnvironmentModel, self).__init__()
        self.opt = opt
        self.ngf = 64
        self.inc = Inconv(4, self.ngf)
        self.downnet = nn.Sequential(
            Down(64, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            Down(128, 256),
        )
        self.action_dim = opt["action_dim"]
        self.action_feature = self.ngf
        self.upnet = nn.Sequential(
            Up(256 + self.action_dim, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            Up(128, 64),
        )
        self.outc = Outconv(self.ngf, 4)
        self.action_fc = nn.Sequential(
            nn.Linear(in_features=self.action_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.action_feature),
            nn.ReLU()
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, current_image_state, action_disc):

        input = self.inc(current_image_state)
        feature = self.downnet(input)
        n_repeat = feature.shape[3]
        action = torch.zeros((action_disc.size()[0], self.opt["action_dim"]))
        for index in range(action_disc.size()[0]):
            action[index, action_disc[index]] = 1
        action = action.float().cuda()
        action_f = action.repeat(n_repeat,n_repeat,1,1).permute(2,3,0,1)

        feature = torch.cat((feature,action_f),1)

        out = self.upnet(feature)
        return self.outc(out)

