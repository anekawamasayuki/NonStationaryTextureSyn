import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import copy
from PIL import Image
import numpy as np
import ntpath
from util import util


# This function is used to testing during training. Results are stored in the opt.results_dir.
# We do not need to run test script again.
def test_func(opt_train, epoch='latest'):
    opt = copy.deepcopy(opt_train)
    print(opt)
    # specify the directory to save the results during training
    opt.results_dir = './results/'
    opt.isTrain = False
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataroot = opt.dataroot + '/test'
    opt.model = 'test'
    opt.dataset_mode = 'single'
    opt.which_epoch = epoch
    opt.how_many = 50
    opt.phase = 'test'
    # opt.name = name

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    # web_dir = os.path.join(opt.results_dir, opt.name)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)

        # visualizer.save_images_epoch(webpage, visuals, img_path, epoch)

        image_dir = opt.results_dir + '/' + opt.name + '/images'
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        for label, image_numpy in visuals.items():
            image_name = '%s_%s_%s.png' % (name, label, epoch)
            save_path = os.path.join(image_dir, image_name)
            print(save_path)
            util.save_image(image_numpy[0].astype(np.uint8), save_path)

    # webpage.save()
