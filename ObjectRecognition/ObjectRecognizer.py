import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as py


class EmptyLayer(nn.Module):
    def __init__(self):
        '''
        The YOLOV3 architecture utilizes "route" layers that are
        pretty much just empty nn.Modules. This is why this class 
        is defined
        '''
        super(EmptyLayer, self).__init__()


class Detector(nn.Module):
    '''
    Likewise, the detection layer is simply just an empty Class that
    stores the anchor tensors.
    '''

    def __init__(self, anchors):
        super(Detector, self).__init__()
        self.anchors = anchors


class ObjectRecognizer(nn.Module):
    def __init__(self):
        '''
        This model implements the YOLOV3 (you only look once v3) object recognition
        architecture to recognize objects. It includes a rudimentary dataloader that
        loads and trains data from CIFAR-100 and COCO as well as a YOLOV3 cfg file 
        parser that parses the file into layers.
        @author Daniel Krivokuca
        @date 2020-03-08
        '''
        super(ObjectRecognizer, self, CUDA=True).__init__()
        self.structure = self.parse_structure_object()
        self.network_info, self.network = self.build_network()
        self.GPU = CUDA

    # overwrite nn.Modules self.forward() function. Since we need the output of the last
    # features mask for our route and shortcut layers we need to store them in a dict and refer
    # to them once we reach those layers
    def forward(self, x):
        layers = self.structure[1:]
        outputs = {}
        collector_initialized = 0
        # iterate over our network layers
        for i, layer in enumerate(layers):
            layer_name = layer['name']

            if(layer_name == "convolutional" or layer_name == "upsample"):
                x = self.network[i](x)

            elif(layer_name == "route"):
                layers = layer["layers"]
                layers = [int(a) for a in layers]

                if(layers[0] > 0):
                    layers[0] = layers[0] - i

                if(len(layers) == 1):
                    x = outputs[i + layer[0]]

                else:
                    if(layers[1] > 0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif(layer_name == "shortcut"):
                from_layer = int(layer['from'])
                x = outputs[i-1] + outputs[i+from_layer]

            elif(layer_name == "yolo"):
                # this is our detection layer
                anchors = self.network[i][0].anchors
                input_dimensions = int(self.network_info["height"])
                num_classes = int(layer['classes'])

                # apply the transformation
                x = x.data
                x = self.feature_map_to_tensor(
                    x, input_dimensions, anchors, num_classes)
                if not collector_initialized:
                    detections = x
                    collector_initialized = 1

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = x

            return detections

    def feature_map_to_tensor(prediction, input_dimensions, anchors, num_classes):
        '''
        This function takes a feature ma and transforms it into a 2d tensor with each
        row of the tensor representing a boundary box and each coordinate of the 
        feature map. For example, given a 7x7 feature map, the output would be a bounding
        box such that Bb_n = (x,y) for n in len(prediction)
        '''
        stride = input_dimensions // prediction.size(2)
        batch_size = prediction.size(0)
        grid_size = input_dimensions // stride
        bounding_box_attributes = 5 + num_classes

        tensor_prediction = prediction.view(
            batch_size, bounding_box_attributes * len(anchors), grid_size * grid_size)
        tensor_prediction = tensor_prediction.transpose(1, 2).contiguous()
        tensor_prediction = tensor_prediction.view(
            batch_size, grid_size*grid_size*len(anchors), bounding_box_attributes)

        # resize the anchors by dividing it by the stride
        n_anchors = len(anchors)
        anchors = [(x[0] / stride, x[1] / stride) for x in anchors]

        # resize our prediction tensor
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

        grid_size = np.arange(grid_size)
        x, y = np.meshgrid(grid_size, grid_size)
        x_offset = torch.FloatTensor(x).view(-1, 1)
        y_offset = torch.FloatTensor(y).view(-1, 1)

        if self.GPU:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        xy_offset = torch.cat((x_offset, y_offset), 1).repeat(
            1, n_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += xy_offset
        anchors = torch.FloatTensor(anchors)

        if self.GPU:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
        prediction[:, :, 5: 5 +
                   num_classes] = torch.sigmoid((prediction[:, :, 5:5 + num_classes]))
        prediction[:, :, :4] *= stride
        return prediction

    def parse_structure_object(self):
        '''
        This functions takes the yolov3 configuration file and parses it into a list
        of dicts where each dictionary corresponds to a layer in the yolov3 config 
        file.
        '''
        structure = []
        with open('./yolov3.cfg.txt', 'r') as f:
            block = {}
            for line in f:
                if line[0] == '#' or line == '' or not line:
                    continue
                line = line.rstrip().lstrip()
                if line and line[0] == '[':
                    if len(block) != 0:
                        structure.append(block)
                    block = {}
                    block['name'] = line.replace('[', '').replace(']', '')
                spl = line.split('=')
                if len(spl) == 2:
                    block[spl[0].strip()] = spl[1].strip()
        return structure

    def build_network(self):
        '''
        This function iterates through the parsed structure objects and constructs an
        nn.Sequential model from each of the layers defined in the structure file. 
        '''
        # first we iterate over the configuration list
        struct_list = self.parse_structure_object()
        input_layer = struct_list[0]

        # the depth of the conv2d layer is the depth of the feature map of the last layer.
        # therefore we need to store the last feature map depth in order to apply it to the
        # next conv layer
        module_list = nn.ModuleList()
        last_fm_depth = 3
        filters = []

        for i, val in enumerate(struct_list[1:]):
            modules = nn.Sequential()

            if(val['name'] == "convolutional"):
                layer_filters = int(val['filters'])
                if 'batch_normalize' in val:
                    batch = int(val['batch_normalize'])
                    has_bias = False
                else:
                    batch = 0
                    has_bias = True
                conv = nn.Conv2d(last_fm_depth, layer_filters, int(
                    val['size']), int(val['stride']), int(val['pad']), bias=has_bias)
                modules.add_module("conv_{}".format(i), conv)
                if not has_bias:
                    batch_layer = nn.BatchNorm2d(layer_filters)
                    modules.add_module("batch_{}".format(i), batch_layer)

                if val['activation'] == "leaky":
                    activation_layer = nn.LeakyReLU(0.1, inplace=True)
                    modules.add_module("leaky_{}".format(i), activation_layer)

            elif(val['name'] == "upsample"):
                upsample_layer = nn.Upsample(mode="bilinear", scale_factor=2)
                modules.add_module("upsample_{}".format(i), upsample_layer)

            elif(val['name'] == "route"):
                route_layers = val['layers'].split(',')
                start = int(route_layers[0])
                if(len(route_layers) > 1):
                    end = int(route_layers[1])
                else:
                    end = 0

                if start > 0:
                    start = start - i

                if end > 0:
                    end = end - i

                route = EmptyLayer()
                modules.add_module("route_{}".format(i), route)

                if end < 0:
                    layer_filters = filters[i + start] + filters[i + end]
                else:
                    layer_filters = filters[i + start]

            elif(val['name'] == "shortcut"):
                shortcut = EmptyLayer()
                modules.add_module("shortcut_{}".format(i), shortcut)

            # the detection layer
            elif(val['name'] == 'yolo'):
                mask = val['mask'].split(',')
                mask = [int(m) for m in mask]
                anchors = val['anchors'].split(',')
                anchors = [int(anchor) for anchor in anchors]
                anchors = [(anchors[i], anchors[i+1])
                           for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection_layer = Detector(anchors)
                modules.add_module("yolo_{}".format(i), detection_layer)

            module_list.append(modules)
            last_fm_depth = layer_filters
            filters.append(layer_filters)
        return(struct_list[0], module_list)
