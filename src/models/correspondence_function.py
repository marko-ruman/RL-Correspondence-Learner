import src.utils.utils as utils
from src.models.correspondence_function_arch import define_Gen, define_Dis


class CorrespondenceFunction:
    def __init__(self, args, q_function=None, environment_model=None):

        self.args = args

        self.q_function = q_function

        self.environment_model = environment_model

        # Define the network
        #####################################################

        image_channels = 4
        if self.args.square_concat:
            image_channels = 1

        self.Gab, A, b = define_Gen(input_nc=image_channels, output_nc=image_channels, ngf=args.ngf, netG=args.gen_net,
                                    norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba, A, b = define_Gen(input_nc=image_channels, output_nc=image_channels, ngf=args.ngf, netG=args.gen_net,
                                    norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids, A=A, b=b)
        self.Da = define_Dis(input_nc=image_channels, ndf=args.ndf, netD= args.dis_net, n_layers_D=args.dis_net_layers,
                             norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=image_channels, ndf=args.ndf, netD= args.dis_net, n_layers_D=args.dis_net_layers,
                             norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db], ['Gab', 'Gba', 'Da', 'Db'])










