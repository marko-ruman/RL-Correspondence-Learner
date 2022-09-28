import functools
import torch
from torch import nn
from .ops import conv_norm_relu, dconv_norm_relu, ResidualBlock, get_norm_layer, init_network
import numpy as np
import torch.nn.functional as F


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                                innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                    conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)
        self.input_nc = input_nc

    def square_concat_orig(self, image):
        image = image.detach().cpu().numpy()[0]
        c = np.concatenate(image[:int(image.shape[0]/2)], axis=1)
        d = np.concatenate([c, np.concatenate(image[int(image.shape[0]/2):], axis=1)])
        d = d*2.0-1
        d = d.reshape((1, 1)+ tuple(d.shape[:]))
        return torch.tensor(d)

    def forward(self, x):
        # if self.input_nc == 1:
        #     x = self.square_concat_orig(x)
        return self.res_model(x)


class AffineResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6, side=None, A=None, b=None):
        super(AffineResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.A = A
        self.b = b
        # if side == "right":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, 0.932, 0], [-0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        # elif side == "left":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, -0.932, 0], [0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        if A is None:
            self.theta = torch.nn.Parameter(torch.tensor(-np.pi/2+0.3, requires_grad=True))

        else:
            self.theta = torch.nn.Parameter(torch.tensor(-A, dtype=torch.float, requires_grad=True))
        self.A = self.theta.clone().detach()
        res_model = [nn.ReflectionPad2d(3),
                     conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.resnet = nn.Sequential(*res_model)
        self.affine = []
        self.affine.append(self.theta)

    def stn(self, x):
        cos_t = torch.cos(self.theta).view(1).cuda()
        sin_t = torch.sin(self.theta).view(1).cuda()
        msin_t = -sin_t.cuda()
        zero = torch.zeros(1).cuda()

        # create rotation matrix using only pytorch functions
        rot_1d = torch.cat((cos_t, sin_t, zero, msin_t, cos_t, zero))
        self.rot_mat = rot_1d.view((1, 2, 3))
        grid1 = F.affine_grid(self.rot_mat, x.size()).cuda()
        x = F.grid_sample(x, grid1,
                          # padding_mode="reflection"
                          )
        return x

    def forward(self, x):
        return self.stn(self.resnet(x))


class TotalAffineResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6, side=None, A=None, b=None):
        super(TotalAffineResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.A = A
        self.b = b
        # if side == "right":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, 0.932, 0], [-0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        # elif side == "left":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, -0.932, 0], [0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        # if A is None:
        #     self.theta = torch.nn.Parameter(torch.tensor(np.pi/2-0.3, requires_grad=True))
        #
        # else:
        #     self.theta = torch.nn.Parameter(torch.tensor(-A, dtype=torch.float, requires_grad=True))
        # self.A = self.theta.clone().detach()


        self.res_model_top = [nn.ReflectionPad2d(3),
                     conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        self.res_model_top = nn.Sequential(*self.res_model_top)

        self.thetas = []
        self.residual_blocks = []
        for i in range(num_blocks):
            self.residual_blocks += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]
            self.thetas.append(torch.nn.Parameter(torch.tensor(np.random.uniform(low=-np.pi, high=np.pi), requires_grad=False)))

        self.thetas.append(torch.nn.Parameter(torch.tensor(np.random.uniform(low=-np.pi, high=np.pi), requires_grad=False)))

        self.res_model_bottom = [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        # self.resnet = nn.Sequential(*res_model)
        self.res_model_bottom = nn.Sequential(*self.res_model_bottom)
        self.resnet = nn.Sequential(*self.res_model_top, *self.residual_blocks, *self.res_model_bottom)
        # self.affine = []
        self.affine = self.thetas

    def stn(self, x, theta):
        cos_t = torch.cos(theta).view(1).cuda()
        sin_t = torch.sin(theta).view(1).cuda()
        msin_t = -sin_t.cuda()
        zero = torch.zeros(1).cuda()

        # create rotation matrix using only pytorch functions
        rot_1d = torch.cat((cos_t, sin_t, zero, msin_t, cos_t, zero))
        rot_mat = rot_1d.view((1, 2, 3))
        grid1 = F.affine_grid(rot_mat, x.size()).cuda()
        x = F.grid_sample(x, grid1, padding_mode="reflection")
        return x

    def forward(self, x):
        x = self.res_model_top(x)
        for i, block in enumerate(self.residual_blocks):
            x = self.stn(x, self.thetas[i])
            x = block(x)
        x = self.stn(x, self.thetas[-1])

        return self.res_model_bottom(x)


class AffineGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6, side=None, A=None, b=None):
        super(AffineGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.b = b
        self.A = A
        # if side == "right":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, 0.932, 0], [-0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        # elif side == "left":
        #     self.theta1 = torch.nn.Parameter(torch.tensor([[[0.362, -0.932, 0], [0.932, 0.362, 0]]], dtype=torch.float, requires_grad=True))
        # # self.theta2 = torch.nn.Parameter(torch.tensor([[[0, 1, 0], [-1, 0, 0]]], dtype=torch.float))
        # self.theta3 = torch.nn.Parameter(torch.tensor([[[0, 1, 0], [-1, 0, 0]]], dtype=torch.float))
        # self.theta4 = torch.nn.Parameter(torch.tensor([[[0, 1, 0], [-1, 0, 0]]], dtype=torch.float))
        # self.theta1 = torch.nn.Parameter(
        #     torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, requires_grad=True))
        if A is None:
            # t = torch.rand(1, 2, 3, dtype=torch.float)
            # self.A = t[0, :2, :2]
            # self.A = self.A/torch.norm(self.A)
            # self.b = torch.tensor([t[0, 0, 2], t[0, 1, 2]])
            # self.b = 0*self.b/torch.norm(self.b)
            # self.theta1 = torch.nn.Parameter(t)
            # self.theta = torch.nn.Parameter(torch.tensor(np.random.uniform(low=-np.pi, high=np.pi), requires_grad=False))
            # self.theta = torch.nn.Parameter(torch.tensor(-np.pi/2, requires_grad=False))
            self.theta = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
            # self.theta1 = torch.nn.Parameter(
            #     torch.tensor([[[self.A[0, 0], self.A[0, 1], self.b[0]], [self.A[1, 0], self.A[1, 1], self.b[1]]]],
            #                  dtype=torch.float, requires_grad=True))
            # self.A = self.A.clone().detach()
            # self.b = self.b.clone().detach()
            self.A = self.theta.clone().detach()
        else:
            # inv_A = np.linalg.inv(A)
            # inv_A_b = -np.dot(inv_A, b)
            # self.A = inv_A
            # self.b = inv_A_b
            # self.theta1 = torch.nn.Parameter(
            #     torch.tensor([[[inv_A[0, 0], inv_A[0, 1], inv_A_b[0]], [inv_A[1, 0], inv_A[1, 1], inv_A_b[1]]]], dtype=torch.float, requires_grad=True))
            self.theta = torch.nn.Parameter(torch.tensor(-A, dtype=torch.float, requires_grad=True))
        # self.lin = nn.Linear(10, 10)

    def stn(self, x):
        cos_t = torch.cos(self.theta).view(1).cuda()
        sin_t = torch.sin(self.theta).view(1).cuda()
        msin_t = -sin_t.cuda()
        zero = torch.zeros(1).cuda()

        # create rotation matrix using only pytorch functions
        rot_1d = torch.cat((cos_t, sin_t, zero, msin_t, cos_t, zero))
        self.rot_mat = rot_1d.view((1, 2, 3))
        grid1 = F.affine_grid(self.rot_mat, x.size()).cuda()
        x = F.grid_sample(x, grid1, padding_mode="reflection")
        return x

    def forward(self, x):

        x = self.stn(x)
        return x



def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0], side=None, A=None, b=None):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6)
    elif netG == 'affine_resnet':
        gen_net = AffineResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                        num_blocks=9, side=side, A=A, b=b)
        A, b = gen_net.A, gen_net.b
    elif netG == 'total_affine_resnet':
        gen_net = TotalAffineResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                        num_blocks=9, side=side, A=A, b=b)
        A, b = gen_net.A, gen_net.b
    elif netG == 'affine':
        gen_net = AffineGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              num_blocks=9, side=side, A=A, b=b)
        A, b = gen_net.A, gen_net.b
    elif netG == 'unet_128':
        gen_net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids), A, b